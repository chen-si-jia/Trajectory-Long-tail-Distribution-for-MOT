cd src

# MOT17_add_our_method_dla34 
# visualThreshold_1.0, strength_sampling_threshold_0.9  gs 3 
CUDA_VISIBLE_DEVICES=0 python train.py mot --load_model '../models/ctdet_coco_dla_2x.pth' \
                            --exp_id mot17_add_our_method_correction_dla34/mot17_add_our_method_correction_visualThreshold_1.0/strength_sampling_threshold_0.9_gs_3 \
                            --data_cfg '../src/lib/cfg/mot17_add_our_method_correction_A_street_or_mall_with_no_pedestrians_visualThreshold_1.0_strength_0.4.json'\
                            --strength_sampling_threshold 0.9 \
                            --id_loss "gs" --gs_group_num 3 \
                            --gs_label2binlabel "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_correction/3分类/label2binlabel.pt" \
                            --gs_pred_slice "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_correction/3分类/pred_slice_with0.pt" \
                            --gs_id_newid "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT17_train_correction/id_newid.txt"


cd ..