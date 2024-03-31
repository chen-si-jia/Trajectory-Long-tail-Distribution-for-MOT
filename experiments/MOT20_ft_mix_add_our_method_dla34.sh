# Note：fairmot_dla34.pth has been trained on mixed data sets.

cd src

# MOT20_ft_mix_add_our_method_correction_dla34 
# visualThreshold_1.0, strength_sampling_threshold_1.0  gs 2 
CUDA_VISIBLE_DEVICES=0 python train.py mot --load_model '../models/fairmot_dla34.pth' --num_epochs 20 --lr_step '15' \
                            --exp_id MOT20_ft_mix_add_our_method_correction_dla34/mot20_add_our_method_correction_visualThreshold_1.0/strength_sampling_threshold_1.0_gs_2 \
                            --data_cfg '../src/lib/cfg/mot20_add_our_method_correction_visualThreshold_1.0.json'\
                            --strength_sampling_threshold 1.0 \
                            --id_loss "gs" --gs_group_num 2 \
                            --gs_label2binlabel "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT20_train_correction/2分类/label2binlabel.pt" \
                            --gs_pred_slice "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT20_train_correction/2分类/pred_slice_with0.pt" \
                            --gs_id_newid "/mnt/A/hust_chensijia/FairMOT/dataset/GS_correction/MOT20_train_correction/id_newid.txt"


cd ..