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
    sam_checkpoint = "/mnt/A/hust_chensijia/segment-anything/WorkSpace/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # 设备选择
    # device = "cpu"  # or  "cuda"
    device = "cuda:0"  # 使用cuda 0 进行推理

    # 设定工作目录为指定目录
    # os.chdir("/mnt/A/hust_csj/Code/segment-anything/WorkSpace")
    # curDirectory = os.getcwd()
    # print("curDirectory_new:", curDirectory)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    Predictor = SamPredictor(sam)

    return Predictor


# -----------------------     -----------------------
def SAMtackle(predictor, backgroundImagePath, originalImagePath, input_box_xywh, tx, ty):
    # -------------------- 开始处理 --------------------
    # 读取图片
    backgroundImageBGR = cv2.imread(backgroundImagePath)
    originalImageBGR = cv2.imread(originalImagePath)

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, backgroundImageBGR)  # 保存编码后的图像文件

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, originalImageBGR)  # 保存编码后的图像文件

    # BGR转为RGB
    originalImageRGB = cv2.cvtColor(originalImageBGR, cv2.COLOR_BGR2RGB)
    predictor.set_image(originalImageRGB) # 编码

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, originalImageRGB)  # 保存编码后的图像文件

    # 输入待抠图的框坐标，格式：[x1,y1,x2,y2]
    # input_box = np.array([1841, 272, 1906, 449]) 
    input_box_xyxy = np.array([input_box_xywh[0], input_box_xywh[1], input_box_xywh[0] + input_box_xywh[2], input_box_xywh[1] + input_box_xywh[3]]) # 图片中的人物框 xyxy

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box_xyxy[None, :],
        multimask_output=False,
    )

    masks1 = masks[0] 
    masks1 = masks1.astype(np.uint8)

    # res：保存目标人物区域，其余区域像素值为0（即为黑色）
    res = cv2.bitwise_and(originalImageBGR, originalImageBGR, mask=masks1)  #mask=mask表示要提取的区域

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, res)  # 保存编码后的图像文件

    # 定义平移矩阵
    # 图像左上角为原点
    # tx = -400  # x轴平移量
    # ty = 50   # y轴平移量
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    # 平移变换
    translated_masks_image = cv2.warpAffine(masks1, M, (masks1.shape[1], masks1.shape[0]))

    # 平移变换 （已保存的目标人物区域）
    translated_image = cv2.warpAffine(res, M, (res.shape[1], res.shape[0]))

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, translated_image)  # 保存编码后的图像文件

    # 目标人物区域像素值为0，其余地方设为1
    # 0 1 互换
    translated_masks_image[translated_masks_image == 1] = 255 
    translated_masks_image[translated_masks_image == 0] = 1 
    translated_masks_image[translated_masks_image == 255] = 0 

    # res1：保存目标人物区域以外的地方，目标人物区域像素值为0（即为黑色）
    res1 = cv2.bitwise_and(backgroundImageBGR, backgroundImageBGR, mask=translated_masks_image)

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, res1)  # 保存编码后的图像文件

    # 两张图片混合得到新输出图片
    output_image = cv2.bitwise_or(translated_image, res1) 

    saveFile = "temp.png"  # 需保存图片文件的路径
    cv2.imwrite(saveFile, output_image)  # 保存编码后的图像文件

    # 输出框 int类型输出
    output_box = np.array([input_box_xywh[0] + tx, input_box_xywh[1] + ty, input_box_xywh[2], input_box_xywh[3]], np.float32)

    return (output_image, output_box) # 以元组形式返回，不能被改变



# 主函数
if __name__ == '__main__':
    
    # 分割器初始化
    samPredictor = SAMinit()

    imagePath = 'WorkSpace/data/images/Circle_View1_000001.jpg' # 读取的图片路径
    input_box = np.array([1841, 272, 65, 177]) # 图片中的人物框 xywh
    # 注意：opencv处理的图片 左上角为原点
    tx = -400  # x轴平移量 
    ty = 50   # y轴平移量

    # 分割器处理
    (output_image, output_box) = SAMtackle(samPredictor, imagePath, imagePath, input_box, tx, ty)

    # 保存已处理的图片
    saveFile = "WorkSpace/data/results/modelProcessed2.png"  # 保存文件的路径
    cv2.imwrite(saveFile, output_image)  # 保存编码后的图像文件
    # 保存新标签
    fo = open("WorkSpace/data/results/modelProcessed2.txt", "w")
    textContent = str(output_box[0]) + "," + str(output_box[1]) + "," + str(output_box[2]) + "," + str(output_box[3])
    fo.write(textContent)
    fo.close()

