# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function
from PIL import Image
from os import path as osp
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pts_utils import generate_label_map
from .file_utils import load_file_lists
from .dataset_utils import pil_loader
from .dataset_utils import anno_parser
from .point_meta import Point_Meta
import torch
import torch.utils.data as data
from .image import get_affine_transform, affine_transform
import cv2


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class GeneralDataset(data.Dataset):

    def __init__(self, transform, sigma, downsample, heatmap_type, data_indicator, phase='train', pca_trans=None):

        self.transform = transform
        self.pca_transform = pca_trans
        self.sigma = sigma
        self.downsample = downsample
        self.heatmap_type = heatmap_type
        self.dataset_name = data_indicator
        self.phase = phase
        self.max_objs = 32
        self.seq_length = 10

        self.reset()
        print('The general dataset initialization done : {:}'.format(self))

    def __repr__(self):
        return (
            '{name}(point-num={NUM_PTS}, sigma={sigma}, heatmap_type={heatmap_type}, length={length}, dataset={dataset_name})'.format(
                name=self.__class__.__name__, **self.__dict__))

    def reset(self, num_pts=-1):
        self.length = 0
        self.NUM_PTS = num_pts
        self.datas = []
        self.labels = []
        self.face_sizes = []
        assert self.dataset_name is not None, 'The dataset name is None'

    def __len__(self):
        assert len(self.datas) == self.length, 'The length is not correct : {}'.format(self.length)
        return self.length

    def append(self, data, labels, box, face_size):
        assert osp.isfile(data[0]), 'The image path is not a file : {}'.format(data)
        self.datas.append(data)
        meat_list = []
        for idx, label in enumerate(labels):
            if (label is not None) and (label.lower() != 'none'):
                if isinstance(label, str):
                    assert osp.isfile(label), 'The annotation path is not a file : {}'.format(label)
                    np_points, _ = anno_parser(label, self.NUM_PTS)
                    meta = Point_Meta(self.NUM_PTS, np_points, box[idx], data[idx], self.dataset_name)
                elif isinstance(label, Point_Meta):
                    meta = label.copy()
                else:
                    raise NameError('Do not know this label : {}'.format(label))
            else:
                meta = Point_Meta(self.NUM_PTS, None, box[idx], data[idx], self.dataset_name)
            meat_list.append(meta)
        self.labels.append(meat_list)
        self.face_sizes.append(face_size)
        self.length = self.length + 1

    def prepare_input(self, image, box):
        meta = Point_Meta(self.NUM_PTS, None, np.array(box), image, self.dataset_name)
        image = pil_loader(image)
        return self._process_(image, meta, -1), meta

    def load_data(self, datas, labels, boxes, face_sizes, num_pts, reset):
        # each data is a png file name
        # each label is a Point_Meta class or the general pts format file (anno_parser_v1)
        assert isinstance(datas, list), 'The type of the datas is not correct : {}'.format(type(datas))
        assert isinstance(labels, list) and len(datas) == len(
            labels), 'The type of the labels is not correct : {}'.format(type(labels))
        assert isinstance(boxes, list) and len(datas) == len(boxes), 'The type of the boxes is not correct : {}'.format(
            type(boxes))
        assert isinstance(face_sizes, list) and len(datas) == len(
            face_sizes), 'The type of the face_sizes is not correct : {}'.format(type(face_sizes))
        if reset:
            self.reset(num_pts)
        else:
            assert self.NUM_PTS == num_pts, 'The number of point is inconsistance : {} vs {}'.format(self.NUM_PTS,
                                                                                                     num_pts)

        print('[GeneralDataset] load-data {:} datas begin'.format(len(datas)))

        for idx, batch_data in enumerate(datas):
            #assert isinstance(data, str), 'The type of data is not correct : {}'.format(data)
            self.append(datas[idx], labels[idx], boxes[idx], face_sizes[idx])

        assert len(self.datas) == self.length, 'The length and the data is not right {} vs {}'.format(self.length,
                                                                                                      len(self.datas))
        assert len(self.labels) == self.length, 'The length and the labels is not right {} vs {}'.format(self.length,
                                                                                                         len(
                                                                                                             self.labels))
        assert len(self.face_sizes) == self.length, 'The length and the face_sizes is not right {} vs {}'.format(
            self.length, len(self.face_sizes))
        print('Load data done for the general dataset, which has {} images.'.format(self.length))


    def load_list(self, file_lists, num_pts, reset):
        lists = load_file_lists(file_lists)
        print('GeneralDataset : load-list : load {:} lines'.format(len(lists)))

        datas, labels, boxes, face_sizes = [], [], [], []
        batch_data, batch_labels, batch_boxes, batch_face_sizes = [], [], [], []
        for idx, data in enumerate(lists):
            alls = [x for x in data.split(' ') if x != '']

            assert len(alls) == 6 or len(alls) == 7, 'The {:04d}-th line in {:} is wrong : {:}'.format(idx, data)
            batch_data.append(alls[0])
            if alls[1] == 'None':
                batch_labels.append(None)
            else:
                batch_labels.append(alls[1])

            box = np.array([float(alls[2]), float(alls[3]), float(alls[4]), float(alls[5])])
            batch_boxes.append(box)
            if len(alls) == 6:
                batch_face_sizes.append(None)
            else:
                batch_face_sizes.append(float(alls[6]))

            if (idx+1)%self.seq_length == 0:
                datas.append(batch_data)
                labels.append(batch_labels)
                boxes.append(batch_boxes)
                face_sizes.append(batch_face_sizes)
                batch_data, batch_labels, batch_boxes, batch_face_sizes = [], [], [], []

        self.load_data(datas, labels, boxes, face_sizes, num_pts, reset)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    #def __getitem__(self, index):
    #    assert index >= 0 and index < self.length, 'Invalid index : {:}'.format(index)
    #    image = pil_loader(self.datas[index])
    #    target = self.labels[index].copy()
    #    return self._process_(image, target, index)

    def __getitem__(self, index, seq_length=10, stride=1):
        assert index >= 0 and index < self.length , 'Invalid index : {:}'.format(index)
        image_list = []
        heatmaps_list = []
        mask_list = []
        points_list = []
        torch_index_list = []
        torch_nopoints_list = []
        ori_size_list = []
        nose_center_hm_list = []
        hp_offset_Lco_list = []
        kps_mask_list = []
        nose_ind_list = []
        for idx in range(seq_length):
            image = pil_loader(self.datas[index][idx])
            target = self.labels[index][idx].copy()
            out = self._process_single_img(image, target, index)
            image, heatmaps, mask, points, torch_index, torch_nopoints, ori_size, nose_center_hm, hp_offset_Lco, kps_mask, nose_ind = out
            image_list.append(image)
            heatmaps_list.append(heatmaps)
            mask_list.append(mask)
            points_list.append(points)
            torch_index_list.append(torch_index)
            torch_nopoints_list.append(torch_nopoints)
            ori_size_list.append(ori_size)
            nose_center_hm_list.append(nose_center_hm)
            hp_offset_Lco_list.append(hp_offset_Lco)
            kps_mask_list.append(kps_mask)
            nose_ind_list.append(nose_ind)
        image = torch.cat(image_list, dim=0)
        heatmaps = torch.cat(heatmaps_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        points = torch.cat(points_list, dim=0)
        torch_index = torch.cat(torch_index_list, dim=0)
        torch_nopoints = torch.cat(torch_nopoints_list, dim=0)
        ori_size = torch.cat(ori_size_list, dim=0)
        nose_center_hm = np.concatenate(nose_center_hm_list, axis=0)
        hp_offset_Lco = np.concatenate(hp_offset_Lco_list, axis=0)
        kps_mask = np.concatenate(kps_mask_list, axis=0)
        nose_ind = np.concatenate(nose_ind_list, axis=0)

        return image, heatmaps, mask, points, torch_index, torch_nopoints, ori_size, nose_center_hm, hp_offset_Lco, kps_mask, nose_ind

    def _process_single_img(self, image, target, index):

        # transform the image and points
        if self.transform is not None:
            image, target = self.transform(image, target)


        # obtain the visiable indicator vector
        if target.is_none():
            nopoints = True
        else:
            nopoints = False

        # If for evaluation not load label, keeps the original data
        temp_save_wh = target.temp_save_wh
        ori_size = torch.IntTensor(
            [temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]])  # H, W, Cropped_[x1,y1]

        if isinstance(image, Image.Image):
            height, width = image.size[1], image.size[0]
        elif isinstance(image, torch.FloatTensor):
            height, width = image.size(1), image.size(2)
        else:
            raise Exception('Unknown type of image : {}'.format(type(image)))

        if target.is_none() == False:
            target.apply_bound(width, height)
            points = target.points.copy()
            points = torch.from_numpy(points.transpose((1, 0))).type(torch.FloatTensor)
            Hpoint = target.points.copy()

            nose_hm = np.zeros((1, height // self.downsample , width // self.downsample), dtype=np.float32)
            hp_offset_Lco = np.zeros((self.max_objs ,self.NUM_PTS*2), dtype=np.float32)
            kps_mask = np.zeros((self.max_objs, self.NUM_PTS * 2), dtype=np.uint8)
            hp_ind = np.zeros((self.max_objs), dtype=np.int64)

        else:
            points = torch.from_numpy(np.zeros((self.NUM_PTS, 3))).type(torch.FloatTensor)
            Hpoint = np.zeros((3, self.NUM_PTS))
            nose_hm = np.zeros((1, height // self.downsample, width // self.downsample), dtype=np.float32)
            hp_offset_Lco = np.zeros((self.max_objs ,self.NUM_PTS * 2), dtype=np.float32)
            kps_mask = np.zeros((self.max_objs, self.NUM_PTS * 2), dtype=np.uint8)
            hp_ind = np.zeros((self.max_objs), dtype=np.int64)
            hp_mask = np.zeros((self.max_objs * self.NUM_PTS), dtype=np.int64)

        heatmaps, mask = generate_label_map(Hpoint, height // self.downsample, width // self.downsample, self.sigma,
                                            self.downsample, nopoints, self.heatmap_type)  # H*W*C

        output_res = 32
        nose_hm = heatmaps[:,:,30][np.newaxis, ...]
        peak_ind = np.argmax(nose_hm)
        row = peak_ind // 32
        col = peak_ind % 32
        ct_int = np.array([row, col]) ##choose nose as face center
        hp_ind[0] = ct_int[1] * output_res + ct_int[0]
        face_bbox_w = (Hpoint[0,:].max() - Hpoint[0,:].min())/4
        face_bbox_h = (Hpoint[1,:].max() - Hpoint[1,:].min())/4
        nose_point_radius = gaussian_radius((math.ceil(face_bbox_h), math.ceil(face_bbox_w)))
        nose_center_hm = draw_umich_gaussian(nose_hm[0], ct_int, min(4, int(nose_point_radius)))
        nose_center = Hpoint[:2, 30].astype(np.int32)

        for j in range(self.NUM_PTS):
            if Hpoint[2, j] > 0:  # means this joint can be seen
                Hpoint[:2, j] = Hpoint[:2, j] // 4
                if Hpoint[0, j] >= 0 and Hpoint[0, j] < output_res and Hpoint[1, j] >= 0 and Hpoint[1, j] < output_res:
                    hp_offset_Lco[0, j * 2: j * 2 + 2] = Hpoint[:2, j] - nose_center
                    kps_mask[0, j * 2: j * 2 + 2] = 1

        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1))).type(torch.FloatTensor)
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))  #####.type(torch.bool)

        torch_index = torch.IntTensor([index])
        torch_nopoints = torch.ByteTensor([nopoints])

        return image, heatmaps, mask, points, torch_index, torch_nopoints, ori_size, nose_center_hm, hp_offset_Lco, kps_mask, hp_ind





    def plot_porttraits(self, images, titles, h, w, n_row, n_col):
        plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def reconstruction(self, weights, C, M, h, w, num_components):
        centered_vector = np.dot(weights[:num_components], C[:num_components, :])
        recovered_image = (M + centered_vector).reshape(h, w)
        return recovered_image


def bgr2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.114, 0.587, 0.299])
