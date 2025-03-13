# 功能：对训练集的标签(det.txt)、新加标签(det_add_label.txt)进行合并
# 撰写人：陈思佳
# 时间：2023-10-16


import os

if __name__ == '__main__':

    # visualThresholdValues = [0.4, 0.5, 0.6, 0.7, 0.8]
    visualThresholdValues = [0.0, 0.1, 0.2, 0.3, 0.9, 1.0]

    for visualThresholdValue in visualThresholdValues:

        # ------------------   参数   ------------------
        sequenceNames = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"]
        
        for sequenceName in sequenceNames:

            # 训练集标签路径
            labelPath1 = "/mnt/A/hust_chensijia/segment-anything/datasets/MOT17/train/" + sequenceName + "/det/"
            filename1 = "det.txt"

            # 新加标签路径
            labelPath2 = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/visualThreshold_" + str(visualThresholdValue) + "/" + sequenceName + "/"
            filename2 = "det_add_label.txt"

            # 处理后结果路径
            tackledsavelabelPath = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/tackled/visualThreshold_" + str(visualThresholdValue) + "/" + sequenceName + "/det/"

            det_txt = []
            det_add_label_txt = []

            num_data = []
            det_final_txt = []

            # 读取 det_train_half 文本文件内容
            txtFile1 = open(os.path.join(labelPath1, filename1),'rb')

            for line in txtFile1.readlines():
                temp = line.strip().split()
                temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
                data = temp[0].split(',') # 以,分割
                det_txt.append(data)

            # 读取 det_add_label 文本文件内容
            txtFile2 = open(os.path.join(labelPath2, filename2),'rb')

            for line in txtFile2.readlines():
                temp = line.strip().split()
                temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
                data = temp[0].split(',') # 以,分割

                data[2] = str(round(float(data[2]), 1))
                data[3] = str(round(float(data[3]), 1))
                data[4] = str(round(float(data[4]), 1))
                data[5] = str(round(float(data[5]), 1))

                det_add_label_txt.append(data)

            # --------------- 开始处理 ---------------
            for i in range(len(det_txt)):
                if det_txt[i][0] not in num_data:
                    for j in range(len(det_add_label_txt)):
                        if (det_add_label_txt[j][0] == det_txt[i][0]):
                            det_final_txt.append(det_add_label_txt[j])

                    num_data.append(det_txt[i][0]) # 将此序号存入累计数字中
                    det_final_txt.append(det_txt[i])
                else:
                    det_final_txt.append(det_txt[i])


            # 写入
            with open(tackledsavelabelPath + "det.txt","w") as f:
                for k in range(len(det_final_txt)):
                    text = det_final_txt[k][0] + "," + det_final_txt[k][1] + "," + det_final_txt[k][2] + "," + det_final_txt[k][3] \
                        + "," + det_final_txt[k][4] + "," + det_final_txt[k][5] + "," + det_final_txt[k][6] + "\n"
                    f.write(text)  # 自带文件关闭功能，不需要再写f.close()

            print("over")

