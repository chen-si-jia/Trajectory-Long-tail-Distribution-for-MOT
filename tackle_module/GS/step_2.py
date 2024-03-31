# Function：Group ids.
# Input：newid_number.txt
# Output：label2binlabel.pt、pred_slice_with0.pt
# Note：Folders need to be created in advance

from lvis.lvis import LVIS
import numpy as np
import pickle
import pdb
import os
import json
import torch
from pycocotools.coco import COCO


def get_cate_gs(GS_num, input_dir_path, save_dir_path):
    train_id = []
    train_num = []
    train_ann_file = input_dir_path + 'newid_number.txt'
    
    file = open(train_ann_file,'rb')
    for line in file.readlines():    
        curLine = line.strip() 
        curLine = str(curLine, 'utf-8')
        curLine = curLine.split(",")

        train_id.append(int(curLine[0])) 
        train_num.append(int(curLine[1])) 

    binlabel_count = []
    for i in range(GS_num):
        binlabel_count.append(1)

    label2binlabel = np.zeros((GS_num, max(train_id) + 1), dtype=np.int32) 

    for i in range(len(train_id)):
        cid = train_id[i]
        ins_count = train_num[i]

        if 2 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            else:
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
        if 3 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            else:
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
        if 4 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            elif ins_count < int(max(train_num)/GS_num*3): 
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
            else:
                label2binlabel[3, cid] = binlabel_count[3]
                binlabel_count[3] += 1
        if 5 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            elif ins_count < int(max(train_num)/GS_num*3): 
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
            elif ins_count < int(max(train_num)/GS_num*4): 
                label2binlabel[3, cid] = binlabel_count[3]
                binlabel_count[3] += 1
            else:
                label2binlabel[4, cid] = binlabel_count[4]
                binlabel_count[4] += 1
        if 6 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            elif ins_count < int(max(train_num)/GS_num*3): 
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
            elif ins_count < int(max(train_num)/GS_num*4): 
                label2binlabel[3, cid] = binlabel_count[3]
                binlabel_count[3] += 1
            elif ins_count < int(max(train_num)/GS_num*5): 
                label2binlabel[4, cid] = binlabel_count[4]
                binlabel_count[4] += 1
            else:
                label2binlabel[5, cid] = binlabel_count[5]
                binlabel_count[5] += 1
        if 7 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            elif ins_count < int(max(train_num)/GS_num*3): 
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
            elif ins_count < int(max(train_num)/GS_num*4): 
                label2binlabel[3, cid] = binlabel_count[3]
                binlabel_count[3] += 1
            elif ins_count < int(max(train_num)/GS_num*5): 
                label2binlabel[4, cid] = binlabel_count[4]
                binlabel_count[4] += 1
            elif ins_count < int(max(train_num)/GS_num*6): 
                label2binlabel[5, cid] = binlabel_count[5]
                binlabel_count[5] += 1
            else:
                label2binlabel[6, cid] = binlabel_count[6]
                binlabel_count[6] += 1
        if 8 == GS_num:
            if ins_count < int(max(train_num)/GS_num): 
                label2binlabel[0, cid] = binlabel_count[0]
                binlabel_count[0] += 1
            elif ins_count < int(max(train_num)/GS_num*2): 
                label2binlabel[1, cid] = binlabel_count[1]
                binlabel_count[1] += 1
            elif ins_count < int(max(train_num)/GS_num*3): 
                label2binlabel[2, cid] = binlabel_count[2]
                binlabel_count[2] += 1
            elif ins_count < int(max(train_num)/GS_num*4): 
                label2binlabel[3, cid] = binlabel_count[3]
                binlabel_count[3] += 1
            elif ins_count < int(max(train_num)/GS_num*5): 
                label2binlabel[4, cid] = binlabel_count[4]
                binlabel_count[4] += 1
            elif ins_count < int(max(train_num)/GS_num*6): 
                label2binlabel[5, cid] = binlabel_count[5]
                binlabel_count[5] += 1
            elif ins_count < int(max(train_num)/GS_num*7): 
                label2binlabel[6, cid] = binlabel_count[6]
                binlabel_count[6] += 1
            else:
                label2binlabel[7, cid] = binlabel_count[7]
                binlabel_count[7] += 1     
        
    savebin = torch.from_numpy(label2binlabel)

    save_path = save_dir_path + 'label2binlabel.pt'
    torch.save(savebin, save_path)

    # start and length
    pred_slice = np.zeros((GS_num, 2), dtype=np.int32) 
    start_idx = 0
    for i, bincount in enumerate(binlabel_count):
        pred_slice[i, 0] = start_idx
        pred_slice[i, 1] = bincount
        start_idx += bincount

    savebin = torch.from_numpy(pred_slice)
    save_path = save_dir_path + 'pred_slice_with0.pt'
    torch.save(savebin, save_path)

    return pred_slice


if __name__ == '__main__':

    GS_num = 5
    input_dir_path = "/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/"
    save_dir_path = "/mnt/A/hust_csj/Code/BalancedGroupSoftmax/mmdetection/tools/GS_data_tackle/data/MOT20_train_LastQuarter_DCVDA_GS_correction/" + str(GS_num) + "分类/"

    get_cate_gs(GS_num, input_dir_path, save_dir_path)

