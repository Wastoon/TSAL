
def parse_eval_info(cur_eval_video_info):
    length_eval_info = len(cur_eval_video_info)
    cur_epoch = int(cur_eval_video_info[0].split(' ')[2].split('-')[1])
    cur_video_auc = float(cur_eval_video_info[-1].split(' ')[-3].split('-')[1].split(',')[0])
    return cur_epoch, cur_video_auc


file = '/home/mry/Desktop/seed-23197-time-31-Aug-at-07-35-19.log'
fr = open(file, 'r')
a = fr.readlines()

real_result_files_line = a[232:]
train_log_length_first = 11
train_log_length_common = 9
eval_log_length_list = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
total_nme_print_length = 1
deliema_line_length = 1

final_block_line_list = []
per_num_length_first = (train_log_length_first + total_nme_print_length + deliema_line_length + sum(eval_log_length_list))
per_num_length_common = (train_log_length_common + total_nme_print_length + deliema_line_length + sum(eval_log_length_list))
epoch_num = 1 + (len(real_result_files_line)-per_num_length_first) // per_num_length_common

log_eval_info_list = []
model_count = 0
for epoch in range(epoch_num):
    if epoch % 10  ==0:
        start = per_num_length_first*model_count + per_num_length_common*(epoch-model_count)
        end = start + per_num_length_first
        all_info_of_curepoch = real_result_files_line[start : end]
        #all_info_of_curepoch = real_result_files_line[epoch * per_num_length_first:epoch * per_num_length_first + per_num_length_first]
        eval_info = all_info_of_curepoch[train_log_length_first:]
        model_count += 1
    else:
        start = per_num_length_first*model_count + per_num_length_common*(epoch-model_count)
        end = start + per_num_length_common
        all_info_of_curepoch = real_result_files_line[start : end]
        eval_info = all_info_of_curepoch[train_log_length_common:]

    cur_epoch_dict = {}
    cur_auc_list = []
    video_info_dict = {}
    vd_epoch = -1
    for eval_video_id in range(len(eval_log_length_list)):
        cur_eval_video_info = eval_info[eval_video_id*eval_log_length_list[eval_video_id]:(eval_video_id+1)*eval_log_length_list[eval_video_id]]
        cur_epoch, cur_auc = parse_eval_info(cur_eval_video_info)
        cur_auc_list.append(cur_auc)
        vd_epoch = cur_epoch
    cur_nme_list = [float(i) for i in eval_info[-2].strip().split(':')[1].split(',')]
    video_info_dict['NME_total'] = cur_nme_list
    video_info_dict['AUC0.08error'] = cur_auc_list
    cur_epoch_dict[vd_epoch] = video_info_dict
    log_eval_info_list.append(cur_epoch_dict)

pass



