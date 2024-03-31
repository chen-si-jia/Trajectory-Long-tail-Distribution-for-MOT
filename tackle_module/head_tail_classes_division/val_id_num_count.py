# Function：head classes and tail classes division

import os
import matplotlib.pyplot as plt


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:           
		os.makedirs(path)       
	else:
		print("----")


if __name__ == '__main__':
    
    seq_list = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-09-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"]
    
    id_list = [[],[],[],[],[],[],[]]
    all_num_list = []
    id_num_list = [[],[],[],[],[],[],[]]

    id_num_flag_list = [[],[],[],[],[],[],[]] 
    seq_order_num_list = [] 
    seq_order_flag_list = [] 
    seq_include_num_list = []

    for idx,seq in enumerate(seq_list):
        # gt.txt path
        seq_gt_txt = "/mnt/A/hust_chensijia/FairMOT/dataset/MOT17/images/train/" + seq + "/gt/gt.txt"
        
        if "MOT17-02-SDP" == seq:
            start_frame = 301
        elif "MOT17-04-SDP" == seq:
            start_frame = 526
        elif "MOT17-05-SDP" == seq:
            start_frame = 419
        elif "MOT17-09-SDP" == seq:
            start_frame = 263
        elif "MOT17-10-SDP" == seq:
            start_frame = 328
        elif "MOT17-11-SDP" == seq:
            start_frame = 451
        elif "MOT17-13-SDP" == seq:
            start_frame = 376
        else:
            print("error")

        gt_txt = open(seq_gt_txt,'rb')
        
        for line in gt_txt.readlines():
            curLine = line.strip()
            curLine = str(curLine, 'utf-8').split(',') 

            frame = int(curLine[0])
            id = int(curLine[1])

            if frame >= start_frame:
                id_list[idx].append(id)

        gt_txt.close()

        ids = set(id_list[idx])

        mkdir('head_tail_classes/' + "115/" + seq)

        file_gt_headclasses = open('head_tail_classes/' + "115/" + seq + '/' + 'gt_headclasses.txt','w')
        file_gt_tailclasses = open('head_tail_classes/' + "115/" + seq + '/' + 'gt_tailclasses.txt','w')

        for id in ids:
            num = id_list[idx].count(id)
            all_num_list.append(num)
            id_num_list[idx].append([id,num])

            tail_classes_flag = 0

            # num=115 division line
            if num > 115: # head classes
                flag = True
                gt_txt = open(seq_gt_txt,'rb')
                for line in gt_txt.readlines():
                    curLine = line.strip()
                    curLine_str = str(curLine, 'utf-8')
                    curLine_split = str(curLine, 'utf-8').split(',')

                    if id == int(curLine_split[1]):
                        file_gt_headclasses.write(curLine_str+'\n')
                gt_txt.close()
            else:
                tail_classes_flag = 1
                gt_txt = open(seq_gt_txt,'rb')
                for line in gt_txt.readlines():
                    curLine = line.strip()
                    curLine_str = str(curLine, 'utf-8')
                    curLine_split = str(curLine, 'utf-8').split(',')

                    if id == int(curLine_split[1]):
                        file_gt_tailclasses.write(curLine_str+'\n')             
                gt_txt.close()

            id_num_flag_list[idx].append([id,num,tail_classes_flag])
            
        file_gt_headclasses.close()
        file_gt_tailclasses.close()

