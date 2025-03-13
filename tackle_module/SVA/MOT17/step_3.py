# ECC_KF_SAM 第二版
# 功能：处理mot17数据集数据，一次性处理完gt文件中的所有id轨迹，得到静态数据增强后的新图片、det_add_label.txt、gt_add_label.txt
# 输入：id_num.txt、id_num_tackle.txt
# 输出：静态数据增强后的新图片、det_add_label.txt、gt_add_label.txt
# 说明：filenames、dir_path 参数需要修改，需要提前建立好 save_figure_path文件夹。与sam.py一起使用。
# 其他：只对静态数据集进行处理，静态数据集序号：02 04 09
# 撰写人：陈思佳
# 日期：2023-10-16


import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import cv2
from math import sin,cos,sqrt # sin,cos的输入是 弧度
import random

from sam import SAMinit, SAMtackle # SAM

def ECC(src, dst, warp_mode = cv2.MOTION_EUCLIDEAN, eps = 1e-5,
        max_iter = 300, scale = None, align = False):
    """Compute the warp matrix from src to dst.

    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    align: bool
        whether to warp affine or perspective transforms to the source image

    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """
    assert src.shape == dst.shape, "the source image must be the same format to the target image!"

    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation = cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)

    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
        else :
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix, None

def AffinePoints(points, warp_matrix, scale = None):
    """Compute the warp matrix from src to dst.

    Parameters
    ----------
    points : array like
        An Nx2 matrix of N points
    warp_matrix : ndarray
        An 2x3 or 3x3 matrix of warp_matrix.
    scale: float or [int, int]
        scale_ratio: float
        scale_x,scale_y: [float, float]
        scale = (image size of ECC) / (image size now)
        if the scale is not None, which means the transition factor in warp matrix must be multiplied

    Returns
    -------
    warped points : ndarray
        Returns an Nx2 matrix of N aligned points
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    assert points.shape[1] == 2, 'points need (x,y) coordinate'

    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            scale = [scale, scale]
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    v = np.ones((points.shape[0], 1))
    points = np.c_[points, v] # [x, y] -> [x, y, 1], Nx3

    aligned_points = warp_matrix @ points.T
    aligned_points = aligned_points[:2, :].T

    return aligned_points.astype(int)

# 卡尔曼滤波和预测
def KF(position_x_measure, position_y_measure, speed_x_measure, speed_y_measure, giveFrameNum, predictFrameNum, img):
    # 图片尺寸维度信息
    (imgHeight, imgWidth, imgChannel) = img.shape

    # 初始化
    # x,y方向的测量值(由非线性空间转到线性空间)
    position_x_measure = position_x_measure
    position_y_measure = position_y_measure
    speed_x_measure = speed_x_measure
    speed_y_measure = speed_y_measure

    # 先验估计值
    position_x_prior_est = []        # x方向位置的先验估计值
    position_y_prior_est = []        # y方向位置的先验估计值
    speed_x_prior_est = []           # x方向速度的先验估计值
    speed_y_prior_est = []           # y方向速度的先验估计值
    
    # 估计值和观测值融合后的最优估计值
    position_x_posterior_est = []    # 根据估计值及当前时刻的观测值融合到一体得到的最优估计值x位置值存入到列表中
    position_y_posterior_est = []    # 根据估计值及当前时刻的观测值融合到一体得到的最优估计值y位置值存入到列表中
    speed_x_posterior_est = []       # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
    speed_y_posterior_est = []       # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中

    delta_t = 1 # 时间差设置为1秒

    # --------------------------- 初始化 -------------------------
    # 用第1帧测量数据初始化
    X0 = np.array([[position_x_measure[0]],[position_y_measure[0]],[speed_x_measure[0]],[speed_y_measure[0]]])
    # 状态估计协方差矩阵P初始化（其实就是初始化最优解的协方差矩阵）
    P = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]) 

    X_posterior = np.array(X0)      # X_posterior表示上一时刻的最优估计值
    P_posterior = np.array(P)       # P_posterior是继续更新最优解的协方差矩阵
    
    # 将初始化后的数据依次送入(即从第2帧速度往里送)
    for i in range(1, predictFrameNum):
        # ------------------- 下面开始进行预测和更新，来回不断的迭代 -------------------------
        # 状态转移矩阵F，上一时刻的状态转移到当前时刻
        F = np.array([[1.0, 0.0, delta_t, 0.0],
                    [0.0, 1.0, 0.0, delta_t],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])      

        # 控制输入矩阵B
        B = np.array([[delta_t*delta_t/2.0, 0.0],
                    [0.0, delta_t*delta_t/2.0],
                    [delta_t, 0.0],
                    [0.0, delta_t]])

        # ---------------------- 预测  -------------------------
        # X_prior = np.dot(F,X_posterior) + np.dot(B,U)            # 使用加速度，X_prior表示根据上一时刻的最优估计值得到当前的估计值  X_posterior表示上一时刻的最优估计值
        X_prior = np.dot(F,X_posterior)                          # 不使用加速度，X_prior表示根据上一时刻的最优估计值得到当前的估计值  X_posterior表示上一时刻的最优估计值

        position_x_prior_est.append(X_prior[0,0])                # 将根据上一时刻计算得到的x方向最优估计位置值添加到列表position_x_prior_est中
        position_y_prior_est.append(X_prior[1,0])                # 将根据上一时刻计算得到的y方向最优估计位置值添加到列表position_y_prior_est中
        speed_x_prior_est.append(X_prior[2,0])                   # 将根据上一时刻计算得到的x方向最优估计速度值添加到列表speed_x_prior_est中
        speed_y_prior_est.append(X_prior[3,0])                   # 将根据上一时刻计算得到的x方向最优估计速度值添加到列表speed_y_prior_est中
        
        # Q:过程噪声的协方差，p(w)~N(0,Q)，噪声来自真实世界中的不确定性，N(0,Q) 表示期望是0，协方差矩阵是Q。Q中的值越小，说明预估的越准确。
        # Q = np.array([[10, 0.0, 0.0, 0.0],
        #               [0.0, 10, 0.0, 0.0],
        #               [0.0, 0.0, 1, 0.0],
        #               [0.0, 0.0, 0.0, 1]]) 
        # # 目前最适配参数1 Q
        # Q = np.array([[0.001, 0.0, 0.0, 0.0],
        #             [0.0, 1, 0.0, 0.0],
        #             [0.0, 0.0, 0.01, 0.0],
        #             [0.0, 0.0, 0.0, 0.1]])         
        # 目前最适配参数2 Q
        Q = np.array([[0.001, 0.0, 0.0, 0.0],
                    [0.0, 1, 0.0, 0.0],
                    [0.0, 0.0, 0.01, 0.0],
                    [0.0, 0.0, 0.0, 0.1]])   
                
        # 计算状态估计协方差矩阵P
        P_prior_1 = np.dot(F,P_posterior)                        # P_posterior是上一时刻最优估计的协方差矩阵  # P_prior_1就为公式中的（F.Pk-1）
        P_prior = np.dot(P_prior_1, F.T) + Q                     # P_prior是得出当前的先验估计协方差矩阵      # Q是过程协方差

        # ------------------- 更新  ------------------------
        # 测量的协方差矩阵R，一般厂家给提供，R中的值越小，说明测量的越准确。
        # R = np.array([[0.001, 0.0, 0.0, 0.0],
        #               [0.0,  0.001, 0.0, 0.0],
        #               [0.0,  0.0,    1, 0.0],
        #               [0.0,  0.0,    0, 1]])
        # # 目前最适配参数1 R
        # R = np.array([[0.01, 0.0, 0.0, 0.0],
        #             [0.0,  10, 0.0, 0.0],
        #             [0.0,  0.0,    0.1, 0.0],
        #             [0.0,  0.0,    0, 1]])
        # 目前最适配参数2 R
        R = np.array([[10, 0.0, 0.0, 0.0],
                    [0.0,  10, 0.0, 0.0],
                    [0.0,  0.0,    0.1, 0.0],
                    [0.0,  0.0,    0, 1]])


        # 状态观测矩阵H（KF,radar,4*4）
        H = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        # 计算卡尔曼增益K
        k1 = np.dot(P_prior, H.T)                                # P_prior是得出当前的先验估计协方差矩阵
        k2 = np.dot(np.dot(H, P_prior), H.T) + R                 # R是测量的协方差矩阵
        K = np.dot(k1, np.linalg.inv(k2))                        # np.linalg.inv()：矩阵求逆   # K就是当前时刻的卡尔曼增益

        if(i < giveFrameNum): # 前giveFrameNum
            # 测量值（数据输入的时候，就进行了线性化处理，从而可以使用线性H矩阵）
            Z_measure = np.array([[position_x_measure[i]],[position_y_measure[i]],[speed_x_measure[i]],[speed_y_measure[i]]])

            X_posterior_1 = Z_measure - np.dot(H, X_prior)           # X_prior表示根据上一时刻的最优估计值得到当前的估计值
            X_posterior = X_prior + np.dot(K, X_posterior_1)         # X_posterior是根据估计值及当前时刻的观测值融合到一体得到的最优估计值
        else: # giveFrameNum之后
            X_posterior = X_prior

        position_x_posterior_est.append(X_posterior[0, 0])       # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x位置值存入到列表中
        position_y_posterior_est.append(X_posterior[1, 0])       # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y位置值存入到列表中
        speed_x_posterior_est.append(X_posterior[2, 0])          # 根据估计值及当前时刻的观测值融合到一体得到的最优估计x速度值存入到列表中
        speed_y_posterior_est.append(X_posterior[3, 0])          # 根据估计值及当前时刻的观测值融合到一体得到的最优估计y速度值存入到列表中

        # 更新状态估计协方差矩阵P     （其实就是继续更新最优解的协方差矩阵）
        P_posterior_1 = np.eye(4) - np.dot(K, H)                 # np.eye(4)返回一个4维数组，对角线上为1，其他地方为0，其实就是一个单位矩阵
        P_posterior = np.dot(P_posterior_1, P_prior)             # P_posterior是继续更新最优解的协方差矩阵  # P_prior是得出的当前的先验估计协方差矩阵

        # 若在predictFrameNum帧内，预测的坐标抵达图像边缘 结束预测
        if(X_posterior[0, 0] < 0 or X_posterior[0, 0] >= imgWidth):
            break
        if(X_posterior[1, 0] < 0 or X_posterior[1, 0] >= imgHeight):
            break

    # 可视化显示
    if True:
        # 一、绘制x-y图
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(position_x_measure[:], position_y_measure[:], color='red', s=1, label="ECC") # 粗度1 所有元素
        plt.plot(position_x_posterior_est, position_y_posterior_est, color='blue', label="ECC_KF")
        # plt.title("track_Id" + str(targetId[i]) + "_Kf_Frame" + strFrameNum)
        plt.legend()  # Add a legend.
        # plt.show()

        # # 二、单独绘制x,y,Vx,Vy图像
        # fig, axs = plt.subplots(2, 2)

        # # axs[0,0].plot(position_x_true, "-", label="位置x_实际值", linewidth=1) 
        # axs[0,0].plot(position_x_measure[1:], "-", label="位置x_测量值", linewidth=1) 
        # axs[0,0].plot(position_x_posterior_est, "-", label="位置x_卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        # axs[0,0].set_title("位置x")
        # axs[0,0].set_xlabel('k') 
        # axs[0,0].legend() 

        # # axs[0,1].plot(position_y_true, "-", label="位置y_实际值", linewidth=1) 
        # axs[0,1].plot(position_y_measure[1:], "-", label="位置y_测量值", linewidth=1) 
        # axs[0,1].plot(position_y_posterior_est, "-", label="位置y_卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)
        # axs[0,1].set_title("位置y")
        # axs[0,1].set_xlabel('k') 
        # axs[0,1].legend() 
        
        # # axs[1,0].plot(speed_x_true, "-", label="速度x_实际值", linewidth=1) 
        # axs[1,0].plot(speed_x_measure[1:], "-", label="速度x_测量值", linewidth=1) 
        # axs[1,0].plot(speed_x_posterior_est, "-", label="速度x_卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1) 
        # axs[1,0].set_title("速度x")
        # axs[1,0].set_xlabel('k') 
        # axs[1,0].legend() 

        # # axs[1,1].plot(speed_y_true, "-", label="速度y_实际值", linewidth=1) 
        # axs[1,1].plot(speed_y_measure[1:], "-", label="速度y_测量值", linewidth=1) 
        # axs[1,1].plot(speed_y_posterior_est, "-", label="速度y_卡尔曼滤波后的值(融合测量值和估计值)", linewidth=1)  
        # axs[1,1].set_title("速度y")
        # axs[1,1].set_xlabel('k') 
        # axs[1,1].legend() 

        # plt.show()
 
        return (position_x_posterior_est, position_y_posterior_est)

# 功能：根据位置计算对应数量的速度
def CalculateSpeedMeasurement(position_measure):
    # ------------- 计算速度测量值 -------------
    speed_measure = []
    num = len(position_measure)

    if num <= 1: # 异常处理条件
        speed_measure.append(0)
    if num >=2:
        for j in range(num):
            if(0 == j):
                speed_measure.append(position_measure[1] - position_measure[0])
            elif(j > num-2):
                speed_measure.append(position_measure[j] - position_measure[j-1])
            else:
                speed_measure.append((position_measure[j+1] - position_measure[j-1]) / 2)

    return speed_measure


if __name__ == '__main__':

    # SAM分割器初始化
    samPredictor = SAMinit()

    # visualThresholdValues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    visualThresholdValues = [0.4, 1.0]

    for visualThresholdValue in visualThresholdValues:
        # -------------- parameter --------------
        # ------------------   参数   ------------------
        sequenceNames = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-09-SDP"]
        
        for sequenceName in sequenceNames:
            # -------------------------------------------- 02 ---------------------------------------------------
            # -------------- parameter --------------
            # video parameter
            dir_path = '/mnt/A/hust_chensijia/segment-anything/datasets/MOT17/train/' + sequenceName + '/' # 源数据
        
            save_figure_path = "results_FairMOT/mot17_train/tackling/visualThreshold_" + str(visualThresholdValue) + "/" + sequenceName + "/" # 修改可视化阈值
            
            id_num_tackle_Path = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/" + sequenceName 
            det_add_label_path = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/visualThreshold_" + str(visualThresholdValue) + "/" + sequenceName + "/det_add_label.txt"
            gt_add_label_path = "/mnt/A/hust_chensijia/segment-anything/results_FairMOT/mot17_train/id_num/visualThreshold_" + str(visualThresholdValue) + "/" + sequenceName + "/gt_add_label.txt"

            # visualThreshold = 0.4 # 卡尔曼滤波随机粘贴  # 修改可视化阈值
            visualThreshold = visualThresholdValue

            # ---- FairMOT ----
            # 训练集的一半
            # frameNum = 300 # 02
            # frameNum = 525 # 04
            # frameNum = 262 # 09
            # ------------------ FairMOT ------------------
            # 整个训练集
            if "MOT17-02-SDP" == sequenceName:
                frameNum = 600 # 
            if "MOT17-04-SDP" == sequenceName:
                frameNum = 1050 # 
            if "MOT17-09-SDP" == sequenceName:
                frameNum = 525 #      
            # ---- Bytetrack ----
            # 训练集的一半
            # frameNum = 301 # 02
            # frameNum = 526 # 04
            # frameNum = 263 # 09
            # 整个训练集
            # frameNum = 600 # 02
            # frameNum = 1050 # 04
            # frameNum = 525 # 09
            # ---------------------------------------

            targetIds = []
            # 创建二维空列表 坐标
            targetIdX = [[] for i in range(4000)]
            targetIdY = [[] for i in range(4000)]
            targetIdW = [[] for i in range(4000)]
            targetIdH = [[] for i in range(4000)]

            # sam copypaste必备
            copyPasteIdConfidence = [[] for i in range(4000)]
            copyPasteIdClass = [[] for i in range(4000)]    
            copyPasteIdVisibility = [[] for i in range(4000)]

            detAddTextContents = [] # 检测新增文本内容
            gtAddTextContents = [] # gt新增文本内容

            # id出现的帧
            targetIdFrame = [[] for i in range(4000)]

            gts = np.genfromtxt(dir_path + 'gt/gt.txt', delimiter=',')
            # 获得该组第一张图片的高和宽
            firstImg = cv2.imread(dir_path + "img1/000001.jpg")
            (imgHeight, imgWidth, imgChannel) = firstImg.shape # 图片尺寸维度信息

            # 读取 gt 文件内容
            labelPath = dir_path + "gt"
            filename = "gt.txt"
            txtFile = open(os.path.join(labelPath, filename),'rb')
            for line in txtFile.readlines():
                temp = line.strip().split()
                temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
                data = temp[0].split(',') # 以,分割
                if(1 == int(data[7]) or 2 == int(data[7])): # 目标属于行人或者在车辆上的人
                    if(int(data[0]) <= frameNum): # 当在目标帧内
                        if int(data[1]) not in targetIds:
                            targetIds.append(int(data[1]))
                            # targetIdbox[int(data[1])].append(int(data[2])) # X
                            # targetIdbox[int(data[1])].append(imgHeight - (int(data[3]) + int(data[5]))) # imgHeight - : 左下角的Y
                            # targetIdbox[int(data[1])].append(int(data[4])) # W
                            # targetIdbox[int(data[1])].append(int(data[5])) # H
                        # 将框的帧数记录下来
                        targetIdFrame[int(data[1])].append(int(data[0]))
                        # 将框的左上角点坐标存起来 左下角为原点
                        targetIdX[int(data[1])].append(int(data[2]))
                        targetIdY[int(data[1])].append(imgHeight - int(data[3])) # imgHeight - ：因为标注的坐标系原点在左上方，绘图的坐标系原点在左下方
                        targetIdW[int(data[1])].append(int(data[4]))
                        targetIdH[int(data[1])].append(int(data[5]))

                        # 存读取的txt文件内容
                        copyPasteIdConfidence[int(data[1])].append(data[6])
                        copyPasteIdClass[int(data[1])].append(data[7])
                        copyPasteIdVisibility[int(data[1])].append(data[8])

            # 读取 id_num_tackle 文件内容
            # 获取 需增强的id的列表
            filename = "id_num_tackle.txt"
            need_enhance_ids = []
            need_enhance_multiple = 0
            txtFile = open(os.path.join(id_num_tackle_Path, filename),'rb')
            for line in txtFile.readlines():
                temp = line.strip().split()
                temp[0] = str(temp[0], 'utf-8') # 将b'1' 转化为 '1'
                data = temp[0].split(',') # 以,分割
                if(0 != int(data[2])): # id属于需增强的id
                    need_enhance_ids.append(int(data[0]))
                    # need_enhance_multiple = int(data[2])
                    # 人为修正：
                    need_enhance_multiple = 1

            # 遍历每一个id
            for i in range(0, len(targetIds)):

                targetId = targetIds[i] # 获取当前目标id
                print("i:", i, " ","targetId:", targetId) 
                # 测试：
                # targetId = 81
                # print("i:", i, " ","targetId:", targetId) 

                # 如果id不属于需要增强的id列表内，直接下一个
                if targetId not in need_enhance_ids:
                    continue

                # 获取帧信息
                F_start = targetIdFrame[targetId][0]
                F_end = max(targetIdFrame[targetId])
                F_total = F_end - F_start + 1
                F_all = frameNum

                # 条件判定
                if(F_end < F_all):
                    print("F_end < F_all")
                    # 采取倒放策略 
                    # 目标人物从[F_start, F_end] 按照倒放转换到 [F_end+1, min(F_all,F_end+F_total)]
                    
                    rightMarginValue = min(F_all,F_end+F_total) # 获取倒放右边界帧值
                    # 遍历需要粘贴的图像的帧下标
                    for FrameIdx1 in range(F_end+1, rightMarginValue, 1):  # [F_end+1, rightMarginValue)
                        # 获得提供id人物的图像的帧下标
                        FrameIdx2 = 2 * F_end - FrameIdx1

                        # 读取需要粘贴的图像（backgroundImage）
                        # 读取提供id人物的图像（originalImage）
                        backgroundImageFrameNum = "{:0>6d}".format(FrameIdx1)
                        backgroundImagePath = dir_path + "img1/" + backgroundImageFrameNum + ".jpg" # 需要粘贴的图像路径
                        originalImageFrameNum = "{:0>6d}".format(FrameIdx2)
                        originalImagePath = dir_path + "img1/" + originalImageFrameNum + ".jpg" # 提供id人物的图像路径

                        # 判定该帧是否已经被处理过了
                        if(True == os.path.exists("/mnt/A/hust_chensijia/segment-anything/" + save_figure_path + backgroundImageFrameNum + ".jpg")):
                            backgroundImagePath = save_figure_path + backgroundImageFrameNum + ".jpg"  # 新的需要粘贴的图像路径

                        # 输入框 图片中的人物框 xywh 左上角为原点
                        input_box = np.array([targetIdX[targetId][FrameIdx2 - F_start], imgHeight - targetIdY[targetId][FrameIdx2 - F_start], \
                                            targetIdW[targetId][FrameIdx2 - F_start], targetIdH[targetId][FrameIdx2 - F_start]])
                        # 偏移量
                        # tx = -800
                        # ty = 300
                        tx = 0
                        ty = 0
                        # 分割器处理
                        (output_image, output_box) = SAMtackle(samPredictor, backgroundImagePath, originalImagePath, input_box, tx, ty)

                        # cv2.imwrite("temp.jpg", output_image)  # 保存编码后的图像文件

                        # 记录新增det标签
                        # <frame>, <-1>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <visibility>
                        detAddTextContent = str(FrameIdx1) + ",-1," + str(output_box[0]) + "," + str(output_box[1]) + "," \
                                            + str(output_box[2]) + "," + str(output_box[3]) + "," + copyPasteIdVisibility[targetId][FrameIdx2 - F_start]
                        detAddTextContents.append(detAddTextContent)

                        # 记录新增gt标签
                        # <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<confidence>,<class>,<visibility>          
                        gtAddTextContent = str(FrameIdx1) + "," + str(targetId) + "," + str(output_box[0]) + "," + str(output_box[1]) + "," + str(output_box[2]) + "," + str(output_box[3]) + "," \
                                        + copyPasteIdConfidence[targetId][FrameIdx2 - F_start] + "," + copyPasteIdClass[targetId][FrameIdx2 - F_start] + "," + copyPasteIdVisibility[targetId][FrameIdx2 - F_start]
                        gtAddTextContents.append(gtAddTextContent)            

                        # 保存已处理的图片
                        saveFile = save_figure_path + backgroundImageFrameNum + ".jpg"  # 保存文件的路径
                        cv2.imwrite(saveFile, output_image)  # 保存编码后的图像文件

                else:
                    print("F_end == F_all")
                    # 采取卡尔曼预测策略
                    # 在[F_start,F_all]帧区间，随机选择可视化程度大于一定值的目标人物按照卡尔曼滤波未越界的预测点随机选择粘贴。粘贴到：[min(1,(F_start-F_total))，F_start-1]

                    leftMarginValue = max(1, F_start-F_total) # 获取卡尔曼预测的左边界帧值

                    # 获取卡尔曼滤波预测位置
                    speed_x_measure = CalculateSpeedMeasurement(targetIdX[targetId][:]) # 由位置测量值得到速度测量值
                    speed_y_measure = CalculateSpeedMeasurement(targetIdY[targetId][:])
                    kfGiveFrameNum = len(targetIdX[targetId][:]) # 获取给定帧长度
                    kfPredictFrameNum = frameNum # 设置预测帧的数目
                    # 卡尔曼滤波和预测
                    (position_x_posterior_est, position_y_posterior_est) = \
                    KF(targetIdX[targetId][:], targetIdY[targetId][:], speed_x_measure, speed_y_measure, kfGiveFrameNum, kfPredictFrameNum, firstImg)

                    # 遍历需要粘贴的图像的帧下标
                    for FrameIdx1 in range(leftMarginValue, F_start, 1):  # [leftMarginValue, F_start)
                        # 在id出现的帧中，随机选择一个 可视化程度大于阈值 的帧：
                        ChooseFlag = True
                        whileTimes = 0
                        while(ChooseFlag):
                            whileTimes += 1
                            randomNum = random.randint(0, len(copyPasteIdVisibility[targetId])-1) # [0, len(copyPasteIdVisibility[targetId]-1)] 随机整数
                            if float(copyPasteIdVisibility[targetId][randomNum]) > visualThreshold: # 可视化程度阈值 visualThreshold
                                FrameIdx2 = F_start + randomNum
                                break
                            # 如果达到限制，就直接开始下一个循环
                            if(whileTimes > 100):
                                FrameIdx2 = F_end
                                break

                        # 读取需要粘贴的图像（backgroundImage）
                        # 读取提供id人物的图像（originalImage）
                        backgroundImageFrameNum = "{:0>6d}".format(FrameIdx1)
                        backgroundImagePath = dir_path + "img1/" + backgroundImageFrameNum + ".jpg" # 需要粘贴的图像路径
                        originalImageFrameNum = "{:0>6d}".format(FrameIdx2)
                        originalImagePath = dir_path + "img1/" + originalImageFrameNum + ".jpg" # 提供id人物的图像路径

                        # 判定该帧是否已经被处理过了
                        if(True == os.path.exists("/mnt/A/hust_chensijia/segment-anything/" + save_figure_path + backgroundImageFrameNum + ".jpg")):
                            backgroundImagePath = save_figure_path + backgroundImageFrameNum + ".jpg"  # 新的需要粘贴的图像路径

                        # 输入框 图片中的人物框 xywh 左上角为原点
                        input_box = np.array([targetIdX[targetId][FrameIdx2 - F_start], imgHeight - targetIdY[targetId][FrameIdx2 - F_start], \
                                            targetIdW[targetId][FrameIdx2 - F_start], targetIdH[targetId][FrameIdx2 - F_start]])
                        
                        # 随机选择一个预测点，并计算偏移量
                        ChooseFlag = True
                        whileTimes = 0 # 记录循环次数
                        while ChooseFlag:
                            whileTimes += 1
                            num_items = len(position_x_posterior_est)
                            random_index = random.randint(0, num_items-1)

                            xRandom = position_x_posterior_est[random_index]
                            yRandom = position_y_posterior_est[random_index]
                        
                            if((xRandom + targetIdW[targetId][FrameIdx2 - F_start] < imgWidth) and (yRandom + targetIdH[targetId][FrameIdx2 - F_start] < imgHeight)): # 限制在画面内
                                ChooseFlag = False
                                # 记录此时的偏移量tx ty
                                tx = xRandom - targetIdX[targetId][FrameIdx2 - F_start]
                                ty = (imgHeight - yRandom) - (imgHeight - targetIdY[targetId][FrameIdx2 - F_start]) # 左上角为原点

                            # 如果达到限制，就直接开始下一个循环
                            if(whileTimes > 100):
                                tx = 0
                                ty = 0
                                break

                        # 分割器处理
                        (output_image, output_box) = SAMtackle(samPredictor, backgroundImagePath, originalImagePath, input_box, tx, ty)

                        # cv2.imwrite("temp.jpg", output_image)  # 保存编码后的图像文件

                        # 记录新增det标签
                        # <frame>, <-1>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <visibility>
                        detAddTextContent = str(FrameIdx1) + ",-1," + str(output_box[0]) + "," + str(output_box[1]) + "," \
                                            + str(output_box[2]) + "," + str(output_box[3]) + "," + copyPasteIdVisibility[targetId][FrameIdx2 - F_start]
                        detAddTextContents.append(detAddTextContent)

                        # 记录新增gt标签
                        # <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<confidence>,<class>,<visibility>          
                        gtAddTextContent = str(FrameIdx1) + "," + str(targetId) + "," + str(output_box[0]) + "," + str(output_box[1]) + "," + str(output_box[2]) + "," + str(output_box[3]) + "," \
                                        + copyPasteIdConfidence[targetId][FrameIdx2 - F_start] + "," + copyPasteIdClass[targetId][FrameIdx2 - F_start] + "," + copyPasteIdVisibility[targetId][FrameIdx2 - F_start]
                        gtAddTextContents.append(gtAddTextContent)                

                        # 保存已处理的图片
                        saveFile = save_figure_path + backgroundImageFrameNum + ".jpg"  # 保存文件的路径
                        cv2.imwrite(saveFile, output_image)  # 保存编码后的图像文件

            # 保存新增det标签
            # 原先的内容会被清空
            with open(det_add_label_path, 'w') as f:
                for j in range(len(detAddTextContents)):
                    f.write(detAddTextContents[j] + "\n")
                f.close()
            # plt.savefig("results/test/track_Id" + str(targetId) + "_Kf_Frame" + strFrameNum + ".jpg")

            # 保存新增gt标签
            # 原先的内容会被清空
            with open(gt_add_label_path, 'w') as f:
                for j in range(len(gtAddTextContents)):
                    f.write(gtAddTextContents[j] + "\n")
                f.close()
            # plt.savefig("results/test/track_Id" + str(targetId) + "_Kf_Frame" + strFrameNum + ".jpg")

            print(" ")

            print("over")

