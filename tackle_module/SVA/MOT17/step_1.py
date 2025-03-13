# 功能：统计某一个数据集中的id及数量，存入id_num.txt文件
# 输入：数据集
# 输出：id_num.txt
# 说明：dir_path、save_path、frameNum 参数需要修改，需要提前建立好文件夹
# 其他：只对静态数据集进行处理，静态数据集序号：02 04 09 
# 撰写人：陈思佳
# 日期：2023-10-16


import os
import cv2
from math import sin,cos,sqrt # sin,cos的输入是 弧度
import random


if __name__ == '__main__':
    # -------------- parameter --------------
    # ------------------   参数   ------------------
    sequenceNames = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"]
    
    for sequenceName in sequenceNames:
        # 序列名称
        # ------------------ FairMOT ------------------
        if "MOT17-02-SDP" == sequenceName:
            frameNum = 600 # 
        if "MOT17-04-SDP" == sequenceName:
            frameNum = 1050 # 
        if "MOT17-09-SDP" == sequenceName:
            frameNum = 525 #      

        dir_path = 'datasets/MOT17/train/' + sequenceName + "/"

        # video parameter
        dir_path = '/mnt/A/hust_chensijia/segment-anything/datasets/MOT17/train/' + sequenceName + '/' # 标签文件路径
        save_path = '/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/' + sequenceName + '/' # txt保存路径
        # ----------------------------------------------

        targetId = []
        # 创建二维列表 默认元素置零
        targetIdNum = [[0] for i in range(4000)]

        # 读取gt文件内容
        labelPath = dir_path + "gt"
        filename = "gt.txt"
        txtFile = open(os.path.join(labelPath, filename),'rb')
        for line in txtFile.readlines():
            temp = line.strip().split()
            temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
            data = temp[0].split(',') # 以,分割
            if(1 == int(data[7]) or 2 == int(data[7])): # 目标属于行人或者在车辆上的人
                if(int(data[0]) <= frameNum): # 当在目标帧内
                    if int(data[1]) not in targetId:
                        targetId.append(int(data[1]))
                        # targetIdbox[int(data[1])].append(int(data[2])) # X
                        # targetIdbox[int(data[1])].append(imgHeight - (int(data[3]) + int(data[5]))) # imgHeight - : 左下角的Y
                        # targetIdbox[int(data[1])].append(int(data[4])) # W
                        # targetIdbox[int(data[1])].append(int(data[5])) # H
                    
                    # 计数当前id帧出现次数
                    targetIdNum[int(data[1])][0] += 1
        
        # 查看统计结果
        print("--------")

        # 将统计结果保存成txt文件
        # 再次运行写入，会自动将txt原本内容清空
        with open(save_path + "id_num.txt","w") as f:
            for i in range(len(targetId)):
                text = str(targetId[i]) + "," + str(targetIdNum[targetId[i]][0]) + "\n"
                f.write(text)  # 自带文件关闭功能，不需要再写f.close()

        # 查看统计结果
        print("----over----")

