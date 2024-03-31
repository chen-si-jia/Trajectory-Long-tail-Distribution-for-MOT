# Function：In the statistics txt file, set a new id number for each id and record the occurrence times.
# Input：id_target_all_nID_MOT17_half_train.txt
# Output：id_newid.txt, newid_number.txt

import os

file_pathname = "/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/MOT20_train_LastQuarter_DCVDA_GS_correction/id_target_all_nID_MOT20_train_mot20LastQuarter"

line_list = []
id_number_list = [] 

i = 0
for file_name in os.listdir(file_pathname):
    f = open("/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/MOT20_train_LastQuarter_DCVDA_GS_correction/id_target_all_nID_MOT20_train_mot20LastQuarter/" + file_name)
    line = f.readline()               
    while line: 
        line_list.append(int(line)) 
        line = f.readline() 
    i += 1
    print("i:", i)

ids = []
for id in set(line_list):
    ids.append(id)

ids.sort() 

for id in ids:
    number = line_list.count(id) // 30 
    id_number_list.append([id,number])

f.close()         

f = open('/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/id_newid.txt','w')
for i in id_number_list:
    f.write(str(i[0])+","+str(id_number_list.index(i)+1)+'\n') # 新id从1开始

f.close() 

f = open('/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/newid_number.txt','w')
for i in id_number_list:
    f.write(str(id_number_list.index(i)+1)+","+str(i[1])+'\n')

f.close() 
