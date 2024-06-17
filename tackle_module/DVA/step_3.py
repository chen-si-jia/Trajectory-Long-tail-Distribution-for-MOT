# 功能：diffusion增强后的合并处理
# 输入：diffusion增强后的图片、原图
# 输出：合并处理的图片
# 说明：sequenceName 参数需要修改，需要提前建立好保存路径的文件夹！
# 其他：只对动态摄像头数据集进行处理，动态数据集序号：05 10 11 13 
# 日期：2023-10-25


import numpy as np
import torch
import cv2
import os
import sys
from segment_anything import sam_model_registry, SamPredictor

def SAMinit():
    # -------------------- SAM初始化 --------------------
    # SAM最小模型
    # sam_checkpoint = "WorkSpace/models/sam_vit_b_01ec64.pth"
    # model_type = "vit_b"

    # SAM最大模型
    sam_checkpoint = "WorkSpace/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # 设备选择
    # device = "cpu"  # or  "cuda"
    device = "cuda:0"  # 使用cuda 3 进行推理

    # 设定工作目录为指定目录
    # os.chdir("/mnt/A/hust_csj/Code/segment-anything/WorkSpace")
    # curDirectory = os.getcwd()
    # print("curDirectory_new:", curDirectory)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    Predictor = SamPredictor(sam)

    return Predictor


def SAMTackleBoxes(predictor, backgroundImagePath, originalImagePath, input_boxes_xywh, tx, ty):
    # -------------------- 开始处理 --------------------
    # 读取图片
    backgroundImageBGR1 = cv2.imread(backgroundImagePath)
    print("backgroundImageBGR1:", backgroundImageBGR1.shape)
    
    originalImageBGR = cv2.imread(originalImagePath)

    backgroundImageBGR = cv2.resize(backgroundImageBGR1, (originalImageBGR.shape[1], originalImageBGR.shape[0])) # resize
    print("backgroundImageBGR:", backgroundImageBGR.shape)

    # saveFile = "temp.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, backgroundImageBGR)  # 保存编码后的图像文件

    # saveFile = "temp.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, originalImageBGR)  # 保存编码后的图像文件

    # BGR转为RGB
    originalImageRGB = cv2.cvtColor(originalImageBGR, cv2.COLOR_BGR2RGB)
    predictor.set_image(originalImageRGB) # 编码

    # saveFile = "temp.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, originalImageRGB)  # 保存编码后的图像文件

    # 输入待抠图的框坐标格式转换，格式：[x1,y1,x2,y2]
    input_boxes_xyxy = []
    for i, box in enumerate(input_boxes_xywh):
        input_boxes_xyxy.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
    
    # numpy -> tensor, cpu -> gpu
    device = torch.device(predictor.device)
    input_boxes_xyxy = torch.Tensor(input_boxes_xyxy).type(torch.int64).to(device)


    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes_xyxy, originalImageBGR.shape[:2])
    # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes_xyxy, [1080, 1920])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # 多个框的时候，masks的第一维度就代表box的数量
    # print("masks.shape:", masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W

    # masks, _, _ = predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_box_xyxy[None, :],
    #     multimask_output=False,
    # )

    # 保存所有掩码至一个矩阵
    for i in range(masks.shape[0]):
        if 0 == i:
            masks1 = np.array(masks[i][0].cpu(), dtype=np.uint8)
        else:
            masks2 = np.array(masks[i][0].cpu(), dtype=np.uint8)
            masks1 = cv2.bitwise_or(masks1, masks2) # 取或

    # masks1 = masks[1] # 0 - masks.shape[0]
    # tensor -> numpy, gpu -> cpu
    # masks1 = masks1.astype(np.uint8)
    # masks1 = np.array(masks1[0].cpu(), dtype=np.uint8)

    # res：保存目标人物区域，其余区域像素值为0（即为黑色）
    res = cv2.bitwise_and(originalImageBGR, originalImageBGR, mask=masks1)  #mask=mask表示要提取的区域
    
    # saveFile = "temp1.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, res)  # 保存编码后的图像文件

    # 定义平移矩阵
    # 图像左上角为原点
    # tx = -400  # x轴平移量
    # ty = 50   # y轴平移量
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    # 平移变换
    translated_masks_image = cv2.warpAffine(masks1, M, (masks1.shape[1], masks1.shape[0]))

    # 平移变换 （已保存的目标人物区域）
    translated_image = cv2.warpAffine(res, M, (res.shape[1], res.shape[0]))

    # saveFile = "temp2.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, translated_image)  # 保存编码后的图像文件

    # 目标人物区域像素值为0，其余地方设为1
    # 0 1 互换
    translated_masks_image[translated_masks_image == 1] = 255 
    translated_masks_image[translated_masks_image == 0] = 1 
    translated_masks_image[translated_masks_image == 255] = 0 

    print("translated_masks_image:", translated_masks_image.shape)
    print("backgroundImageBGR:", backgroundImageBGR.shape)

    # res1：保存目标人物区域以外的地方，目标人物区域像素值为0（即为黑色）
    res1 = cv2.bitwise_and(backgroundImageBGR, backgroundImageBGR, mask=translated_masks_image)

    # saveFile = "temp3.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, res1)  # 保存编码后的图像文件

    # 两张图片混合得到新输出图片
    output_image = cv2.bitwise_or(translated_image, res1) 

    # saveFile = "temp4.png"  # 需保存图片文件的路径
    # cv2.imwrite(saveFile, output_image)  # 保存编码后的图像文件

    return output_image


# 主函数
if __name__ == '__main__':
    
    # 分割器初始化
    samPredictor = SAMinit()

    # ------------------   参数   ------------------
    sequenceNames = ["MOT17-05-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"]

    for sequenceName in sequenceNames:

        # 序列名称
        # sequenceName = "MOT17-11-SDP"

        if "MOT17-11-SDP" == sequenceName:
            promptNames = ["A_mall_with_no_pedestrians"] 
        else:
            promptNames = ["A_street_with_no_pedestrians"] 

            # promptNames = ["none_input", "A_street_with_no_pedestrians", "An_empty_street", "A_crowded_street", "A_street"]
            # promptNames = ["none_input"] # 
            # promptNames = ["An_empty_street"]

        for promptName in promptNames:

            # ------------------ FairMOT ------------------
            # MOT17_train
            if "MOT17-05-SDP" == sequenceName:
                frameNumAll = 837 # 05
            if "MOT17-10-SDP" == sequenceName:
                frameNumAll = 654 # 10
            if "MOT17-11-SDP" == sequenceName:
                frameNumAll = 900 # 11
            if "MOT17-13-SDP" == sequenceName:
                frameNumAll = 750 # 13


            # 参数
            dir_path = 'datasets/MOT17/train/' + sequenceName + '/'

            # 创建二维空列表 坐标
            targetFrame = []
            targetFrameX = [[] for i in range(frameNumAll + 1)]
            targetFrameY = [[] for i in range(frameNumAll + 1)]
            targetFrameW = [[] for i in range(frameNumAll + 1)]
            targetFrameH = [[] for i in range(frameNumAll + 1)]
            targetFrameXYWH = [[] for i in range(frameNumAll + 1)]

            gts = np.genfromtxt(dir_path + 'gt/gt.txt', delimiter=',')
            # 获得该组第一张图片的高和宽
            img = cv2.imread(dir_path + "img1/000001.jpg")
            (imgHeight, imgWidth, imgChannel) = img.shape # 图片尺寸维度信息

            # 读取 gt 文件内容
            labelPath = dir_path + "gt"
            filename = "gt.txt"
            txtFile = open(os.path.join(labelPath, filename),'rb')
            for line in txtFile.readlines():
                temp = line.strip().split()
                temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
                data = temp[0].split(',') # 以,分割
                if(1 == int(data[7]) or 2 == int(data[7])): # 目标属于行人或者在车辆上的人
                    frameNum = int(data[0])
                    if(frameNum <= frameNumAll): # 当在目标帧内
                        # 存放帧数
                        if frameNum not in targetFrame:
                            targetFrame.append(frameNum) 
                        # 将框的左上角点坐标存起来 左下角为原点
                        targetFrameX[frameNum].append(int(data[2]))
                        targetFrameY[frameNum].append(int(data[3]))
                        targetFrameW[frameNum].append(int(data[4]))
                        targetFrameH[frameNum].append(int(data[5]))

                        targetFrameXYWH[frameNum].append([int(data[2]), int(data[3]), int(data[4]), int(data[5])])


            for i,frame in enumerate(targetFrame):

                print("frame:", frame)

                input_boxes = targetFrameXYWH[frame]
                input_boxes = np.array(input_boxes)

                # 读取的图片路径
                backgroundImagePath = "./results_FairMOT_2/mot17_train/tackling/diffusion/" + sequenceName + "/" + promptName + "/" + "{:0>6d}".format(frame) + ".jpg"
                originalImagePath = "./datasets/MOT17/train/" + sequenceName + "/img1/" + "{:0>6d}".format(frame) + ".jpg"

                # 保存的图像路径
                # 用绝对路径准没错，相对路径有时候会出错
                saveFilePath = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT_2/mot17_train/tacked/merge/" + sequenceName + "/" + promptName + "/" + "{:0>6d}".format(frame) + ".jpg"

                # 注意：opencv处理的图片 左上角为原点
                tx = 0  # x轴平移量
                ty = 0   # y轴平移量

                # 分割器处理
                output_image = SAMTackleBoxes(samPredictor, backgroundImagePath, originalImagePath, input_boxes, tx, ty)

                # 保存已处理的图片
                # saveFile = "WorkSpace/data/results/modelProcessed2.png"  # 保存文件的路径
                # saveFile = "temp.png"  # 保存文件的路径
                cv2.imwrite(saveFilePath, output_image)  # 保存编码后的图像文件

                # print(input_boxes)


