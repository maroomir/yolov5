import json
import time
from pathlib import Path

import cv2.cv2
import numpy
import torch.cuda
from numpy import ndarray

from utils.general import xyxy2xywh, xywh2xyxy
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


class Fusing:
    def __init__(self,
                 path: str,
                 dist_offset=100,  # pixel
                 size_offset=0.2,  # rate
                 picks=['person']
                 ):
        with open(path, "r") as fp:
            data = json.load(fp)
            types = {"main_h": int, "main_w": int, "sub_h": int, "sub_w": int, "cx": int, "cy": int,
                     "dx": int, "dy": int, "rw": float, "rh": float}
            self.cam_param = [_type(data[key]) for key, _type in types.items()]
        self.main_img, self.main_boxes, self.main_lbs = None, None, None
        self.sub_img, self.sub_boxes, self.sub_lbs = None, None, None
        self.dist_offset = dist_offset
        self.size_offset = size_offset
        self.picks = picks
        self.flag = [False, False]

    def input_main(self, im: ndarray, xyxy, lbs):
        if len(xyxy) > 0 or len(lbs) > 0:
            xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
            lbs = [lbs] if not isinstance(lbs, list) else lbs
            self.main_img, self.main_lbs = im, lbs
            self.main_boxes = xyxy2xywh(torch.Tensor(xyxy))
            self.flag[0] = True

    def input_sub(self, im: ndarray, xyxy, lbs):
        if len(xyxy) > 0:
            xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
            self.sub_img, self.sub_lbs = im, lbs
            self.sub_boxes = xyxy2xywh(torch.Tensor(xyxy))
            self.flag[1] = True

    def do(self):
        if self.flag != [True, True]:
            return numpy.ascontiguousarray(self.main_img), [], [], [], []
        self.flag = [False, False]

        def bbox_sync(obbox: ndarray, cparam=self.cam_param):
            dx, dy, rw, rh = cparam[6], cparam[7], cparam[8], cparam[9]
            return (obbox[0] * rw) + dx, (obbox[1] * rh) + dy, int(obbox[2] * rw), int(obbox[3] * rh)

        def similarity(src_w, src_h, obj_w, obj_h):
            res_w = (obj_w / src_w) if (src_w >= obj_w) else (src_w / obj_w)
            res_h = (obj_h / src_h) if (src_h >= obj_h) else (src_h / obj_h)
            return res_w, res_h

        def mix_image(src, obj, cparam=self.cam_param, weight=0.5):
            cx, cy, rw, rh = cparam[4], cparam[5], cparam[8], cparam[9]
            t_mat = numpy.float32([[1, 0, cx],
                                   [0, 1, cy]])
            cp_obj = cv2.resize(obj, None, fx=rw, fy=rh)
            obj_h, obj_w = cp_obj.shape[0:2]
            src_h, src_w = src.shape[0:2]
            dh, dw = src_h - obj_h, src_w - obj_w
            cp_obj = cv2.warpAffine(cp_obj, t_mat, (obj_w, obj_h))
            cp_obj = cv2.copyMakeBorder(cp_obj, int(dh/2), int(dh/2), int(dw/2), int(dw/2), cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            res = cv2.addWeighted(src, weight, cp_obj, 1 - weight, 0)
            return res

        matches = []
        match_lbs = []
        miss = []
        miss_lbs = []
        res_image = mix_image(self.main_img, self.sub_img)
        if res_image is None:
            return numpy.ascontiguousarray(self.main_img), [], [], [], []
        for sbbox, main_lb in zip(self.main_boxes, self.main_lbs):
            if not main_lb in self.picks:
                continue
            match = False
            srx, sry, srw, srh = sbbox[0], sbbox[1], sbbox[2], sbbox[3]
            for obbox, sub_lb in zip(self.sub_boxes, self.sub_lbs):
                obx, oby, obw, obh = bbox_sync(obbox)
                dist_x, dist_y = abs(srx - obx), abs(sry - oby)
                sim_w, sim_h = similarity(srw, srh, obw, obh)
                res_x, res_y = (srx + obx) / 2, (sry + oby) / 2
                res_w, res_h = (srw + obw) / 2, (srh + obh) / 2
                res_xywh = torch.Tensor([res_x, res_y, res_w, res_h])
                res_xywh = torch.unsqueeze(res_xywh, dim=0)
                res_xyxy = xywh2xyxy(res_xywh).squeeze()
                if (dist_x <= self.dist_offset) and (dist_y <= self.dist_offset) and\
                        (sim_w <= self.size_offset) and (sim_h <= self.size_offset):
                    matches.append(res_xyxy.tolist())
                    match_lbs.append(main_lb)
                    match = True
                else:
                    miss.append(res_xyxy.tolist())
                    miss_lbs.append(sub_lb)
            if not match:
                src_xywh = torch.Tensor([srx, sry, srw, srh])
                src_xywh = torch.unsqueeze(src_xywh, dim=0)
                src_xyxy = xywh2xyxy(src_xywh).squeeze()
                miss.append(src_xyxy.tolist())
                miss_lbs.append(main_lb)
        res_image = numpy.ascontiguousarray(res_image)
        return res_image, matches, match_lbs, miss, miss_lbs


def get_diff(main_img: ndarray, main_box, sub_img: ndarray, sub_box,
             save_diff=True,
             save_path=ROOT / 'data/theeye/cal_data.json'):
    main_h, main_w, _ = main_img.shape
    sub_h, sub_w, _ = sub_img.shape
    main_xywh = xyxy2xywh(torch.tensor(main_box).view(1, 4)).tolist()
    sub_xywh = xyxy2xywh(torch.tensor(sub_box).view(1, 4)).tolist()
    rw, rh = float(main_w / sub_w), float(main_h / sub_h)
    sub_xywh = [[rw * sub_xywh[0][0], rh * sub_xywh[0][1], rw * sub_xywh[0][2], rh * sub_xywh[0][3]]]
    dx, dy = int(main_xywh[0][0] - sub_xywh[0][0]), int(main_xywh[0][1] - sub_xywh[0][1])
    res = {"main_h": main_h, "main_w": main_w, "sub_h": sub_h, "sub_w": sub_w,
           "dx": dx, "dy": dy, "rw": rw, "rh": rh}
    if save_diff:
        with open(save_path, "w") as fp:
            json.dump(res, fp)
    return res
