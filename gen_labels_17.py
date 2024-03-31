# Note: Each time you run, you need to delete the labels_with_ids folder.

import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

seq_root = '/mnt/A/hust_chensijia/FairMOT/dataset/MOT17_all_strengthened_gs/A_street_or_mall_with_no_pedestrians/visualThreshold_0.1_strength_0.4/MOT17/images/train'
label_root = '/mnt/A/hust_chensijia/FairMOT/dataset/MOT17_all_strengthened_gs/A_street_or_mall_with_no_pedestrians/visualThreshold_0.1_strength_0.4/MOT17/labels_with_ids/train'



mkdirs(label_root)

seqs = ["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-09-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"]

tid_curr = 0
tid_last = -1
for seq in seqs:

    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
