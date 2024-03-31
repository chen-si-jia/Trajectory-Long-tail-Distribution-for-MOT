cd src

CUDA_VISIBLE_DEVICES=0 python track_half.py mot \
                        --exp_name ablation_study_track \
                        --load_model /mnt/A/hust_chensijia/FairMOT/models/fairmot_dla34.pth \
                        --conf_thres 0.4 \
                        --val_mot17 True \
                        --benchmark_name val_mot17


cd ..