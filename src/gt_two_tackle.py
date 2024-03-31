# Function：Sort the gt.txt file twice

import numpy as np

# num_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_list = [1]
for num in num_list:
    seq_list = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"]
    for seq in seq_list:
        gt_path = "/mnt/A/hust_chensijia/FairMOT/dataset/MOT17_all_strengthened_gs/A_street_or_mall_with_no_pedestrians/visualThreshold_0.8_strength_0.4/MOT17/images/train/" + seq + "/gt"

        file_name = '/gt.txt'

        sorted_data = np.genfromtxt(gt_path + file_name, delimiter=',')

        sorted_data = sorted(sorted_data, key=lambda x: (x[1], x[0]))

        output_file = '/gt_tackled.txt'
        np.savetxt(gt_path + output_file, sorted_data, delimiter=',', fmt='%d')




