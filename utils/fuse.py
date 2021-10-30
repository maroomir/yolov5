import json
import time
from pathlib import Path
from threading import Thread

import cv2.cv2
import numpy
from numpy import ndarray

from utils.general import xyxy2xywh

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


class FusingThread(Thread):
    def __init__(self,
                 path: str,
                 dist_offset=10,  # pixel
                 size_offset=0.9,  # rate
                 picks=['person']
                 ):
        super.__init__()
        with open(path, "r") as fp:
            data = json.load(fp)
            types = {"src_h": int, "src_w": int, "obj_h": int, "obj_w": int, "dx": int, "dy": int, "rw": float,
                     "rh": float}
            self.cparam = [type(data[key]) for key, _type in types.items()]
        self.flags = [False, False, False]
        self.src, self.src_boxes, self.labels = None, None, None
        self.obj, self.obj_boxes = None, None
        self.result = []
        self.d_offset = dist_offset
        self.s_offset = size_offset
        self.picks = picks

    def input_src(self, im: ndarray, xyxy, lbs):
        while self.flags[2]:
            time.sleep(0.001)
        xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
        lbs = [lbs] if not isinstance(lbs, list) else lbs
        self.src, self.labels = im, lbs
        self.src_boxes = [xyxy2xywh(ps) for ps in xyxy]
        self.flags[0] = True

    def input_obj(self, im: ndarray, xyxy):
        while self.flags[2]:
            time.sleep(0.001)
        xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
        self.obj = im
        self.src_boxes = [xyxy2xywh(ps) for ps in xyxy]
        self.flags[1] = True

    def run(self):
        def check_param(src, obj):
            src_h, src_w, _ = src.shape
            obj_h, obj_w, _ = obj.shape
            if src_h != self.cparam[0] or src_w != self.cparam[1] or obj_h != self.cparam[2] or obj_w != self.cparam[3]:
                org_rw, org_rh = self.cparam[6], self.cparam[7]
                new_rw, new_rh = float(src_w / obj_w), float(src_h / obj_h)
                org_dx, org_dy = self.cparam[4], self.cparam[5]
                new_dx, new_dy = int(new_rw / org_rw * org_dx), int(new_rh / org_rh * org_dy)
                return [src_h, src_w, obj_h, obj_w, new_dx, new_dy, new_rw, new_rh]
            else:
                return self.cparam

        def bbox_sync(obbox: ndarray, cparam=self.cparam):
            dx, dy, rw, rh = cparam[4], cparam[5], cparam[6], cparam[7]
            return obbox[0] + dx, obbox[1] + dy, int(obbox[2] * rw), int(obbox[3] * rh)

        def similarity(src_w, src_h, obj_w, obj_h):
            res_w = (obj_w / src_w) if (src_w >= obj_w) else (src_w / obj_w)
            res_h = (obj_h / src_h) if (src_h >= obj_h) else (src_h / obj_h)
            return res_w, res_h

        while True:
            if self.flags == [True, True, False]:
                src_im, obj_im = self.src, self.obj
                self.cparam = check_param(src_im, obj_im)
                for sbbox, lb in zip(self.src_boxes, self.labels):
                    if not lb in self.picks:
                        continue
                    for obbox in self.obj_boxes:
                        srx, sry, srw, srh = sbbox[0], sbbox[1], sbbox[2], sbbox[3]
                        obx, oby, obw, obh = bbox_sync(obbox)
                        dist_x, dist_y = abs(srx - obx), abs(sry - oby)
                        sim_w, sim_h = similarity(srw, srh, obw, obh)
                        if (dist_x and dist_y <= self.d_offset) and (sim_w and sim_h <= self.s_offset):
                            res_x, res_y = (srx + obx) / 2, (sry + oby) / 2
                            res_w, res_h = (srw + obw) / 2, (srh + obh) / 2
                            res_xyxy = xyxy2xywh(numpy.int([res_x, res_y, res_w, res_h]))
                            self.res_boxes.append(res_xyxy)
                self.flags = [False, False, True]
                break

    def result(self):
        while True:
            if self.flags == [False, False, True]:
                res_boxes = self.result.copy()
                self.src, self.src_boxes, self.labels = None, None, None
                self.obj, self.obj_boxes = None, None
                self.result.clear()
                self.flags[2] = False
                return res_boxes


def get_diff(src_img: ndarray, src_box, obj_img: ndarray, obj_box,
             save_diff=True,
             save_path=ROOT / 'data/theeye'):
    src_h, src_w, _ = src_img.shape
    obj_h, obj_w, _ = obj_img.shape
    src_xywh, obj_xywh = xyxy2xywh(src_box).tolist(), xyxy2xywh(obj_box).tolist()
    rw, rh = float(src_w / obj_w), float(src_h / obj_h)
    obj_xywh = [rw * obj_xywh[0], rh * obj_xywh[1], rw * obj_xywh[2], rh * obj_xywh[3]]
    dx, dy = int(src_xywh[0] - obj_xywh[0]), int(src_xywh[1] - obj_xywh[1])
    res = {"src_h": src_h, "src_w": src_w, "obj_h": obj_h, "obj_w": obj_w,
           "dx": dx, "dy": dy, "rw": rw, "rh": rh}
    if save_diff:
        with open(save_path, "w") as fp:
            json.dump(res, fp)
    return res
