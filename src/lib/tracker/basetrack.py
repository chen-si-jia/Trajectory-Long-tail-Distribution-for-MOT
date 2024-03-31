import numpy as np
from collections import OrderedDict
import cv2

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed



# 6幅合并图：原图+5张图
def vis_feature(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["00000","0000","000","00","0"]
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M) 
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    dst_path = out_path.vis_img_root + "/" + seq_num + "_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)


def vis_feature_two_figure(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["00000","0000","000","00","0"]
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128 
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M) 
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128 
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1) 

    dst_path = out_path.vis_img_root + "/" + seq_num + "_two_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)


def vis_feature_one_figure(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["00000","0000","000","00","0"] 
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M) 
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*128
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = img 

    dst_path = out_path.vis_img_root + "/" + seq_num + "_one_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)
