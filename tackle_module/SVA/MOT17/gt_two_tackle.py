# 功能：对gt.txt文件进行二次排序，采用numpy
# 撰写人：陈思佳
# 日期：2023-10-16


import numpy as np

# num_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_list = [0.0]
# num_list = [1]
for num in num_list:
    seq_list = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"]
    # seq_list = ["MOT16-02", "MOT16-04", "MOT16-09"]
    for seq in seq_list:
        # gt_path = "/mnt/A/hust_chensijia/FairMOT/dataset/MOT17_still_strengthened/visualThreshold_" + str(num) + "/MOT17/images/train/" + seq + "/gt"
        # gt_path = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot16_train/tackled/visualThreshold_1.0/" + seq + "/gt"
        gt_path = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train_fronthalf/tackled/visualThreshold_" + str(num) + "/" + seq + "/gt"

        # 文件名
        file_name = '/gt.txt'

        # 读取CSV文件并保存为二维数组
        sorted_data = np.genfromtxt(gt_path + file_name, delimiter=',')

        # 对每一行按照规定排序
        sorted_data = sorted(sorted_data, key=lambda x: (x[1], x[0]))

        output_file = '/gt处理完成.txt'
        np.savetxt(gt_path + output_file, sorted_data, delimiter=',', fmt='%d')




