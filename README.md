# Delving into the Trajectory Long-tail Distribution for Muti-object Tracking

> [**【CVPR 2024】Delving into the Trajectory Long-tail Distribution for Muti-object Tracking**](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Delving_into_the_Trajectory_Long-tail_Distribution_for_Muti-object_Tracking_CVPR_2024_paper.html)            
> Sijia Chen, En Yu, Jinyang Li, Wenbing Tao      
> *[ArXiv] Paper ([http://arxiv.org/abs/2403.04700](http://arxiv.org/abs/2403.04700))*  
> *[CVPR] Paper ([Delving_into_the_Trajectory_Long-tail_Distribution_for_Muti-object_Tracking_CVPR_2024_paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Delving_into_the_Trajectory_Long-tail_Distribution_for_Muti-object_Tracking_CVPR_2024_paper.pdf)))*  
> *YouTube ([https://www.youtube.com/watch?v=ohgIesSNgaQ](https://www.youtube.com/watch?v=ohgIesSNgaQ))*   

If you have any problems with our work, please issue me. We will promptly reply it.

If you cite our method for experimental comparison, you can use the method name **TLTDMOT**.

Thanks for your attention! If you are interested in our work, please give us a star ⭐️.

## Poster
![](assets/poster.png)


## Abstract
Multiple Object Tracking (MOT) is a critical area within computer vision, with a broad spectrum of practical implementations. Current research has primarily focused on the development of tracking algorithms and enhancement of post-processing techniques. Yet, there has been a lack of thorough examination concerning the nature of tracking data it self. In this study, we pioneer an exploration into the distribution patterns of tracking data and identify a pronounced long-tail distribution issue within existing MOT datasets. We note a significant imbalance in the distribution of trajectory lengths across different pedestrians, a phenomenon we refer to as “pedestrians trajectory long-tail distribution”. Addressing this challenge, we introduce a bespoke strategy designed to mitigate the effects of this skewed distribution. Specifically, we propose two data augmentation strategies, including Stationary Camera View Data Augmentation (SVA) and Dynamic Camera View Data Augmentation (DVA) , designed for viewpoint states and the Group Softmax (GS) module for Re-ID. SVA is to backtrack and predict the pedestrian trajectory of tail classes, and DVA is to use diffusion model to change the background of the scene. GS divides the pedestrians into unrelated groups and performs softmax operation on each group individually. Our proposed strategies can be integrated into numerous existing tracking systems, and extensive experimentation validates the efficacy of our method in reducing the influence of long-tail distribution on multi-object tracking performance. 


## Apology letter
I'm Sijia Chen. I'm very sorry. There is a small error in Figure 1 in the paper of official CVPR. Figure 1 in the paper of ArXiv is correct. 

We made a mistake when submitting the camera-ready version of CVPR. Although we found this error in May 2024 and contacted the publisher immediately, we were unable to correct it because the deadline for the camera-ready version of CVPR had passed.


## News
* (2025.03.13) The code of SVA is opened! Now, our code for all modules is open!
* (2024.06.17) The code of DVA is opened!
* (2024.04.19) Our poster is selected for the 3th China3DV presentation!
* (2024.02.27) The code of GS is opened!
* (2024.02.27) Our paper is accepted by CVPR 2024!


## Installation
* Note: We use a NVIDIA GeForce RTX 3090 GPU and cuda 11.1.
* Clone this repo, and we'll call the directory that you cloned as ${Trajectory-Long-tail-Distribution-for-MOT_ROOT}
* Install dependencies. We use python 3.8 and pytorch >= 1.7.0
```
conda create -n TLDTMOT python=3.8
conda activate TLDTMOT
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
cd ${Trajectory-Long-tail-Distribution-for-MOT_ROOT}
pip install cython # Optional addition: -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r requirements.txt # Optional addition: -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
* We use [DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7) in our backbone network (pytorch_1.7 branch). Previous versions can be found in [DCNv2](https://github.com/CharlesShang/DCNv2).
```
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
```
* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).
```
conda install ffmpeg
pip install ffmpy
```

## Data preparation
* **2DMOT15 , MOT16, MOT17 and MOT20** 
[2DMOT15](https://motchallenge.net/data/2D_MOT_2015/), [MOT16](https://motchallenge.net/data/MOT16/), [MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT20/) can be downloaded from the official webpage of MOT challenge. The After downloading, you should prepare the data in the following structure:
```
dataset
   |
   |
   |——————MOT15
   |        |——————images
   |        |        └——————train
   |        |        └——————test
   |        └——————labels_with_ids
   |                 └——————train(empty)
   |——————MOT16
   |        |——————images
   |        |        └——————train
   |        |        └——————test
   |        └——————labels_with_ids
   |                 └——————train(empty)
   |——————MOT17
   |        |——————images
   |        |        └——————train
   |        |        └——————test
   |        └——————labels_with_ids
   |                 └——————train(empty)
   |——————MOT20
            |——————images
            |        └——————train
            |        └——————test
            └——————labels_with_ids
                     └——————train(empty)
```
Then, you can change the seq_root and label_root in src/gen_labels_15.py , src/gen_labels_16.py, src/gen_labels_17.py and src/gen_labels_20.py and run:
```
cd src
python gen_labels_15.py
python gen_labels_16.py
python gen_labels_17.py
python gen_labels_20.py
```
to generate the labels of 2DMOT15 , MOT16, MOT17 and MOT20. The seqinfo.ini files of 2DMOT15 can be downloaded here [[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w), [[Baidu],code:8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g).

Note: Each time you run, you need to delete the labels_with_ids folder.

* **CrowdHuman**
The CrowdHuman dataset can be downloaded from their [official webpage](https://www.crowdhuman.org). After downloading, you should prepare the data in the following structure:
```
dataset
   |
   |
   |——————crowdhuman
            |——————images
            |        └——————train
            |        └——————val
            └——————labels_with_ids
            |         └——————train(empty)
            |         └——————val(empty)
            └------annotation_train.odgt
            └------annotation_val.odgt
```
If you want to pretrain on CrowdHuman (we train Re-ID on CrowdHuman), you can change the paths in src/gen_labels_crowd_id.py and run:
```
cd src
python gen_labels_crowd_id.py
```
If you want to add CrowdHuman to the MIX dataset (we do not train Re-ID on CrowdHuman), you can change the paths in src/gen_labels_crowd_det.py and run:
```
cd src
python gen_labels_crowd_det.py
```
* **MIX**
We use the same training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) in this part and we call it "MIX". Please refer to their [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) to download and prepare all the training data including Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16. 

## Pretrained models and baseline model
* **Pretrained models**

DLA-34 official COCO pretrained model: ctdet_coco_dla_2x.pth can be downloaded here [[Baidu, code:hust]](https://pan.baidu.com/s/1r4c9gZCTYMF4mzP0uXQG5g?pwd=hust), [[Google]](https://drive.google.com/file/d/10_lybGkTxhkRMc9oEGK94ccxB-9wGzC_/view?usp=drive_link).
HRNetV2 ImageNet pretrained model: [HRNetV2-W18 official](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw), [HRNetV2-W32 official](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w).
After downloading, you should put the pretrained models in the following structure:
```
${Trajectory-Long-tail-Distribution-for-MOT_ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
           └——————hrnetv2_w32_imagenet_pretrained.pth
           └——————hrnetv2_w18_imagenet_pretrained.pth
```
* **Baseline model**

Our baseline FairMOT model (DLA-34 backbone) is pretrained on the CrowdHuman for 60 epochs with the self-supervised learning approach and then trained on the MIX dataset for 30 epochs. The models can be downloaded here: 
crowdhuman_dla34.pth [[Google]](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [[Baidu, code:ggzx ]](https://pan.baidu.com/s/1JZMCVDyQnQCa5veO73YaMw) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EUsj0hkTNuhKkj9bo9kE7ZsBpmHvqDz6DylPQPhm94Y08w?e=3OF4XN).
fairmot_dla34.pth [[Google]](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?usp=sharing) [[Baidu, code:uouv]](https://pan.baidu.com/s/1H1Zp8wrTKDk20_DSPAeEkg) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EWHN_RQA08BDoEce_qFW-ogBNUsb0jnxG3pNS3DJ7I8NmQ?e=p0Pul1). 
After downloading, you should put the baseline model in the following structure:
```
${Trajectory-Long-tail-Distribution-for-MOT_ROOT}
   └——————models
           └——————fairmot_dla34.pth
           └——————...
```

## The important notes:
Our processed MOT17 dataset by SVA and DVA can be downloaded here [[Google]](https://drive.google.com/drive/folders/15rAPb50yX_HTGOqrevD1VYbEKj1B2349?usp=sharing) [[Baidu, code:hust](https://pan.baidu.com/s/1JnqT4p3cmhTNcj3rhKm__Q?pwd=hust)].

Our models can be downloaded here [[Google]](https://drive.google.com/drive/folders/1Y3t9XQsLgVup-nEprr0XUzY6A4y-NYsR?usp=sharing) [[Baidu, code:hust](https://pan.baidu.com/s/101oi6K6SjC4T6PiQvUh6_A?pwd=hust)].

## Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Only train on MOT15:

Baseline(+Ours):
```
bash experiments/MOT15_add_our_method_dla34.sh
```
Baseline:
```
bash experiments/MOT15_baseline.sh
```
* Only train on MOT16:

Baseline(+Ours):
```
bash experiments/MOT16_add_our_method_dla34.sh
```
Baseline:
```
bash experiments/MOT16_baseline.sh
```
* Only train on MOT17:

Baseline(+Ours):
```
bash experiments/MOT17_add_our_method_dla34.sh
```
Baseline:
```
bash experiments/MOT17_baseline.sh
```

* Only Train on MOT20:

The data annotation of MOT20 is a little different from MOT17, the coordinates of the bounding boxes are all inside the image, so we need to uncomment line 313 to 316 in the dataset file src/lib/datasets/dataset/jde.py:
```
#np.clip(xy[:, 0], 0, width, out=xy[:, 0])
#np.clip(xy[:, 2], 0, width, out=xy[:, 2])
#np.clip(xy[:, 1], 0, height, out=xy[:, 1])
#np.clip(xy[:, 3], 0, height, out=xy[:, 3])
```
Then, we can train on MOT20:

Baseline(+Ours):
```
bash experiments/MOT20_add_our_method_dla34.sh
```
Baseline:
```
bash experiments/MOT20_baseline.sh
```

* Train on MIX and MOT20:
The data annotation of MOT20 is a little different from MOT17, the coordinates of the bounding boxes are all inside the image, so we need to uncomment line 313 to 316 in the dataset file src/lib/datasets/dataset/jde.py:
```
#np.clip(xy[:, 0], 0, width, out=xy[:, 0])
#np.clip(xy[:, 2], 0, width, out=xy[:, 2])
#np.clip(xy[:, 1], 0, height, out=xy[:, 1])
#np.clip(xy[:, 3], 0, height, out=xy[:, 3])
```
Then, we can train on MOT20:

Baseline(+Ours):
```
bash experiments/MOT20_ft_mix_add_our_method_dla34.sh
```

* For ablation study, 

```
bash experiments/ablation_study.sh
```

## Tracking
* To get the txt results of the test set of MOT15 or MOT16 or MOT17 or MOT20, you should modify the '--load_model' in the sh file and run it:

MOT15:
```
bash experiments/MOT15_track.sh
```
MOT16:
```
bash experiments/MOT16_track.sh
```
MOT17:
```
bash experiments/MOT17_track.sh
```
MOT20:
```
bash experiments/MOT20_track.sh
```

* For ablation study

we evaluate on the other half of the training set of MOT17, you can run:

All classes(default):
```
bash experiments/ablation_study_track.sh
```

If you want to evaluate head classes and tail classes, you need to run tackle_module/head_tail_classes_division/val_id_num_count.py. Then you need to place the generated gt_headclasses.txt and gt_tailclasses.txt file in the corresponding gt location of the MOT17 training dataset, like below:
```
dataset
   |
   |
   |——————MOT17
            |
            |——————images
                     |
                     |——————train
                              |
                              |——————MOT17-02-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-04-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-05-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-09-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-10-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-11-SDP
                              |            |
                              |            |——————gt
                              |                   └——————gt.txt
                              |                   └——————gt_headclasses.txt
                              |                   └——————gt_tailclasses.txt
                              |——————MOT17-13-SDP
                                           |
                                           |——————gt
                                                  └——————gt.txt
                                                  └——————gt_headclasses.txt
                                                  └——————gt_tailclasses.txt
```

Then you can run:

Head classes or tail classes:
```
bash experiments/ablation_study_classes_track.sh
```


## Demo
You can input a raw video and get the demo video by running src/demo.py and get the mp4 format of the demo video:
```
cd src
python demo.py mot --load_model ../models/Table_4_MIX_model_20.pth --conf_thres 0.4
```
You can change --input-video and --output-root to get the demos of your own videos.
--conf_thres can be set from 0.3 to 0.7 depending on your own videos.

Note: Table_4_MIX_model_20.pth can be downloaded here [[Google]](https://drive.google.com/file/d/1NFQtDe3PgziaLAXG1dP7mxUO2DP12oU6/view?usp=sharing) [[Baidu, code:hust](https://pan.baidu.com/s/1CjA8s9djQJi97EywzoJ9Eg?pwd=hust)].


## Acknowledgement
The part of the code are borrowed from the follow work:
- [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)
- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

Thanks for their wonderful works.

## Citation
```
@InProceedings{Chen_2024_CVPR,
    author    = {Chen, Sijia and Yu, En and Li, Jinyang and Tao, Wenbing},
    title     = {Delving into the Trajectory Long-tail Distribution for Muti-object Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19341-19351}
}
```
