cd src
  
# MOT15_baseline
CUDA_VISIBLE_DEVICES=0 python train.py mot --load_model '../models/ctdet_coco_dla_2x.pth' \
                            --exp_id mot15_baseline --data_cfg '../src/lib/cfg/mot15.json'

cd ..