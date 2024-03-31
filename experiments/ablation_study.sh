cd src

CUDA_VISIBLE_DEVICES=2 python train.py mot \
                        --exp_id ablation_study/visualThreshold_0.9_strength_0.4/0.9/3 \
                        --load_model '../models/ctdet_coco_dla_2x.pth' \
                        --data_cfg '../src/lib/cfg/Mot17_half_MOT17_all_strengthened_visualThreshold_0.9_strength_0.4.json' \
                        --strength_sampling_threshold 0.9\
                        --id_loss "gs" --gs_group_num 3 \
                        --gs_label2binlabel "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_half_strengthened_after_correction/3分类/label2binlabel.pt" \
                        --gs_pred_slice "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_half_strengthened_after_correction/3分类/pred_slice_with0.pt" \
                        --gs_id_newid "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_half_strengthened_after_correction/id_newid.txt"  


cd ..