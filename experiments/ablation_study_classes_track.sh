cd src

# gt_headclasses track
CUDA_VISIBLE_DEVICES=1 python track_half.py mot --val_mot17 True --exp_name fairmot_dla34_headclasses \
                    --load_model /mnt/A/hust_chensijia/FairMOT/models/fairmot_dla34.pth  --conf_thres 0.4 \
                    --benchmark_name val_mot17 \
                    --gt_txt gt_headclasses.txt

# gt_tailclasses track
CUDA_VISIBLE_DEVICES=1 python track_half.py mot --val_mot17 True --exp_name fairmot_dla34_tailclasses \
                    --load_model /mnt/A/hust_chensijia/FairMOT/models/fairmot_dla34.pth  --conf_thres 0.4 \
                    --benchmark_name val_mot17 \
                    --gt_txt gt_tailclasses.txt



cd ..