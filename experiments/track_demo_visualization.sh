
# ----- visualization -----
cd src

time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3

CUDA_VISIBLE_DEVICES=0 python demo.py mot --load_model ../models/fairmot_dla34.pth  --input-video "" --output-root "" --conf_thres 0.4


cd ..

