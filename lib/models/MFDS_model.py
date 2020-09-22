import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from easydict import EasyDict
from lib.procedure.losses import compute_stage_loss_MFDS

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

class Correlation(nn.Module):
    def __init__(self, max_displacement=4, *args, **kwargs):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.output_dim = 2 * self.max_displacement + 1
        self.pad_size = self.max_displacement

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = F.pad(x2, [self.pad_size] * 4)
        cv = []
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)

def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask
def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad_2nd(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid
def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)
def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()
def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()

class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg, detection_model):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        self.detection_model = detection_model

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []
        recons_multi_scale_img1_loss = []
        recons_multi_scale_img2_loss = []
        unsupervisied_target_by_tracking = torch.zeros_like(im1_origin)

        s = 1.

        criterion_MFDS = torch.nn.MSELoss().cuda()

        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

            if i == 0:
                unsupervisied_target_by_tracking = im2_recons



            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)


            ##consistency between tracking source and detection source
            for p in self.detection_model.parameters():
                p.requires_grad = False

            batch_heatmaps_recons2, batch_locs_recons2, batch_scos_recons2, CLOR_dict_recons2 = self.detection_model(
                im2_recons)
            batch_heatmaps_recons1, batch_locs_recons1, batch_scos_recons1, CLOR_dict_recons1 = self.detection_model(
                im1_recons)
            batch_heatmaps_org2, batch_locs_org2, batch_scos_org2, CLOR_dict_org2 = self.detection_model(
                im2_scaled)
            batch_heatmaps_org1, batch_locs_org1, batch_scos_org1, CLOR_dict_org1 = self.detection_model(
                im1_scaled)

            ##loss of consistency
            FW_total_loss_singlescale, FW_stage_loss_singlescale = compute_stage_loss_MFDS(criterion_MFDS,
                                                                                           batch_heatmaps_org2,
                                                                                           batch_heatmaps_recons2)
            BW_total_loss_singlescale, BW_stage_loss_singlescale = compute_stage_loss_MFDS(criterion_MFDS,
                                                                                           batch_heatmaps_org1,
                                                                                           batch_heatmaps_recons1)
            recons_multi_scale_img2_loss.append(FW_total_loss_singlescale)
            recons_multi_scale_img1_loss.append(BW_total_loss_singlescale)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]

        pyramid_FW_consistency_loss = [l * w for l, w in
                               zip(recons_multi_scale_img2_loss, self.cfg.w_scales)]
        pyramid_BW_consistency_loss = [l * w for l, w in
                                       zip(recons_multi_scale_img1_loss, self.cfg.w_scales)]
        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        FW_consistency_loss = sum(pyramid_FW_consistency_loss)
        BW_consistency_loss = sum(pyramid_BW_consistency_loss)

        total_loss = warp_loss + smooth_loss + FW_consistency_loss + BW_consistency_loss

        return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean(), FW_consistency_loss, BW_consistency_loss, unsupervisied_target_by_tracking.detach()

class MFDS(nn.Module):
    def __init__(self, cfg):
        super(MFDS, self).__init__()
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.upsample = cfg.model.upsample
        self.n_frames = cfg.model.n_frames
        self.reduce_dense = cfg.model.reduce_dense

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + (self.dim_corr + 2) * (self.n_frames - 1)

        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(
            (self.flow_estimators.feat_dim + 2) * (self.n_frames - 1))

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward_2_frames(self, x1_pyramid, x2_pyramid):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            x_intm, flow_res = self.flow_estimators(
                torch.cat([out_corr_relu, x1_1by1, flow], dim=1))
            flow = flow + flow_res

            flow_fine = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            flows.append(flow)

            # upsampling or post-processing
            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward_3_frames(self, x0_pyramid, x1_pyramid, x2_pyramid):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 4, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()

        for l, (x0, x1, x2) in enumerate(zip(x0_pyramid, x1_pyramid, x2_pyramid)):
            # warping
            if l == 0:
                x0_warp = x0
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                x0_warp = flow_warp(x0, flow[:, :2])
                x2_warp = flow_warp(x2, flow[:, 2:])

            # correlation
            corr_10, corr_12 = self.corr(x1, x0_warp), self.corr(x1, x2_warp)
            corr_relu_10, corr_relu_12 = self.leakyRELU(corr_10), self.leakyRELU(corr_12)

            # concat and estimate flow
            x1_1by1 = self.conv_1x1[l](x1)
            feat_10 = [x1_1by1, corr_relu_10, corr_relu_12, flow[:, :2], -flow[:, 2:]]
            feat_12 = [x1_1by1, corr_relu_12, corr_relu_10, flow[:, 2:], -flow[:, :2]]
            x_intm_10, flow_res_10 = self.flow_estimators(torch.cat(feat_10, dim=1))
            x_intm_12, flow_res_12 = self.flow_estimators(torch.cat(feat_12, dim=1))
            flow_res = torch.cat([flow_res_10, flow_res_12], dim=1)
            flow = flow + flow_res

            feat_10 = [x_intm_10, x_intm_12, flow[:, :2], -flow[:, 2:]]
            feat_12 = [x_intm_12, x_intm_10, flow[:, 2:], -flow[:, :2]]
            flow_res_10 = self.context_networks(torch.cat(feat_10, dim=1))
            flow_res_12 = self.context_networks(torch.cat(feat_12, dim=1))
            flow_res = torch.cat([flow_res_10, flow_res_12], dim=1)
            flow = flow + flow_res

            flows.append(flow)

            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4,
                                   mode='bilinear', align_corners=True) for flow in flows]

        flows_10 = [flo[:, :2] for flo in flows[::-1]]
        flows_12 = [flo[:, 2:] for flo in flows[::-1]]
        return flows_10, flows_12

    def forward(self, x, with_bk=False):
        n_frames = x.size(1) / 3

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        res_dict = {}
        if n_frames == 2:
            res_dict['flows_fw'] = self.forward_2_frames(x[0], x[1])
            if with_bk:
                res_dict['flows_bw'] = self.forward_2_frames(x[1], x[0])
        elif n_frames == 3:
            flows_10, flows_12 = self.forward_3_frames(x[0], x[1], x[2])
            res_dict['flows_fw'], res_dict['flows_bw'] = flows_12, flows_10
        elif n_frames == 5:
            flows_10, flows_12 = self.forward_3_frames(x[0], x[1], x[2])
            flows_21, flows_23 = self.forward_3_frames(x[1], x[2], x[3])
            res_dict['flows_fw'] = [flows_12, flows_23]
            if with_bk:
                flows_32, flows_34 = self.forward_3_frames(x[2], x[3], x[4])
                res_dict['flows_bw'] = [flows_21, flows_32]
        else:
            raise NotImplementedError
        return res_dict

if __name__=='__main__':
    with open('/home/mry/PycharmProjects/SALD/configs/MFDS.json') as f:
        cfg = EasyDict(json.load(f))
    model = MFDS(cfg).cuda()
    #img1_py = [torch.rand(4,3,512,512), torch.rand(4,3,256,256), torch.rand(4,3,128,128),torch.rand(4,3,64,64)]
    #img2_py = [torch.rand(4,3,512,512), torch.rand(4,3,256,256), torch.rand(4,3,128,128),torch.rand(4,3,64,64)]

    # read data to device
    img1, img2 =  torch.rand(5,3,256,256),  torch.rand(5,3,256,256)
    img_pair = torch.cat([img1, img2], 1).cuda()
    det_net = nn.Conv2d(2, 3, 3).cuda()

    res_dict = model(img_pair, with_bk=True)
    from thop import profile
    flop, params = profile(model, inputs=(img_pair, ))
    print('flop:{}, params:{}'.format(flop, params))
    # compute output
    flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
    flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
             zip(flows_12, flows_21)]
    cfg = cfg.loss
    loss_func = unFlowLoss(cfg, det_net)
    loss, l_ph, l_sm, flow_mean = loss_func(flows, img_pair)

    pass