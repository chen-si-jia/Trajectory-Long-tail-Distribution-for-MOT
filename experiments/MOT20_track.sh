cd src

CUDA_VISIBLE_DEVICES=1 python track.py mot \
                        --exp_name MOT20_add_our_method_dla34/ \
                        --load_model /mnt/A/hust_chensijia/FairMOT/models/fairmot_dla34.pth \
                        --conf_thres 0.5 \
                        --test_mot20 True \
                        --benchmark_name test_mot20


cd ..