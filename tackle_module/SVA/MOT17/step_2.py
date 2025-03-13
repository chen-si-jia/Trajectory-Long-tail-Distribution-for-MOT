# 功能：划分出少数类多数类，存入id_num_tackle.txt文件
# 输入：id_num.txt
# 输出：id_num_tackle.txt
# 说明：filenames、dir_path 参数需要修改，需要提前建立好文件夹
# 其他：只对静态数据集进行处理，静态数据集序号：02 04 09 
# 撰写人：陈思佳
# 日期：2023-10-16


import os
import cv2
import random
import math

# RPS采样策略
def data_prepare(targetIdNums, rt): # rt：少数类和多数类 划分阈值。rt越大，划分线越趋向于0
    
    tg = []
    # prepare re-sampling factor for category
    for targetIdNum in targetIdNums:
        img_num_cls = targetIdNum
        img_num_total = sum(targetIdNums)
        f = img_num_cls / img_num_total
        t = 0.3
        rc = max(1, math.pow(t / f, 0.5))

        # rt = 5 # 少数类和多数类 划分阈值 
        p = 1 # 增强次数

        if (rc >= rt):
            tg.append(p)
        else:
            tg.append(0)
    
    return tg


if __name__ == '__main__':

    # 标签文件夹路径
    filenames = ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-09-SDP']
    dir_path = '/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/'
    save_path = dir_path

    # 目标id 和 目标id数目 和 每个视频的目标id数
    targetId = [] 
    targetIdNum = []
    targetGroupIdCount = []

    # 遍历所有的文件夹
    for filename in filenames:
        filepath = dir_path + filename # 标签文件路径

        # 文件路径
        filename = "id_num.txt"
        txtFile = open(os.path.join(filepath, filename),'rb')

        # 获取txt文件总行数
        count = len(open(os.path.join(filepath, filename),'rb').readlines())
        print(count)
        targetGroupIdCount.append(count)

        for line in txtFile.readlines():
            temp = line.strip().split()
            temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
            data = temp[0].split(',') # 以,分割

            targetId.append(data[0]) # 以str格式存入
            targetIdNum.append(int(data[1])) # 以int格式存入
    
    # 方法一：对所有视频数据进行一起处理
    # tgs = data_prepare(targetIdNum)

    
    # 方法二：对所有视频数据进行单独处理 
    tgs = []
    for i in range(len(targetGroupIdCount)):
        startNum = 0
        for j in range(i): # 计算startNum
            startNum += targetGroupIdCount[j] 
        if 0 == i: # 02 
            tg = data_prepare(targetIdNum[startNum: startNum+targetGroupIdCount[i]], rt=6) # rt越小，划分线越趋向于0
        if 1 == i: # 04 
            tg = data_prepare(targetIdNum[startNum: startNum+targetGroupIdCount[i]], rt=6)
        if 2 == i: # 09 
            tg = data_prepare(targetIdNum[startNum: startNum+targetGroupIdCount[i]], rt=6)
        tgs.append(tg)

    # 将tgs 从二维列表转为一维列表
    tgs = [i for item in tgs for i in item]

    # 处理分割线
    print("--------")

    # 遍历所有的文件夹
    for i, filename in enumerate(filenames):
        filepath = save_path + filename # 标签文件路径

        startNum = 0
        for j in range(i): # 计算startNum
            startNum += targetGroupIdCount[j]

        # 将统计结果保存成txt文件
        # 再次运行写入，会自动将txt原本内容清空
        with open(filepath + "/" + "id_num_tackle.txt","w") as f:
            for i in range(startNum, startNum+targetGroupIdCount[i]):
                text = str(targetId[i]) + "," + str(targetIdNum[i]) + "," + str(tgs[i]) + "\n"
                f.write(text)  # 自带文件关闭功能，不需要再写f.close()
                
    # 程序结束
    print("----over----")

