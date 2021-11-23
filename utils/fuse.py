import json
import time
from pathlib import Path

import cv2.cv2
import numpy
import torch.cuda
from numpy import ndarray

from utils.general import xyxy2xywh, xywh2xyxy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


class Fusing:
    def __init__(self,
                 path: str,
                 dist_offset=10,  # pixel
                 size_offset=0.9,  # rate
                 picks=['person']
                 ):
        with open(path, "r") as fp:
            data = json.load(fp)
            types = {"main_h": int, "main_w": int, "sub_h": int, "sub_w": int, "dx": int, "dy": int, "rw": float,
                     "rh": float}
            self.cam_param = [_type(data[key]) for key, _type in types.items()]
        self.main_img, self.main_boxes, self.labels = None, None, None
        self.sub_img, self.sub_boxes = None, None
        self.dist_offset = dist_offset
        self.size_offset = size_offset
        self.picks = picks
        self.flag = [False, False]

    def input_main(self, im: ndarray, xyxy, lbs):
        if len(xyxy) > 0 or len(lbs) > 0:
            xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
            lbs = [lbs] if not isinstance(lbs, list) else lbs
            self.main_img, self.labels = im, lbs
            self.main_boxes = xyxy2xywh(torch.Tensor(xyxy))
            self.flag[0] = True

    def input_sub(self, im: ndarray, xyxy):
        if len(xyxy) > 0:
            xyxy = [xyxy] if not isinstance(xyxy, list) else xyxy
            self.sub_img = im
            self.sub_boxes = xyxy2xywh(torch.Tensor(xyxy))
            self.flag[1] = True

    def do(self):
        if self.flag != [True, True]:
            return []
        self.flag = [False, False]

        def check_param(src, obj):
            src_h, src_w, _ = src.shape
            obj_h, obj_w, _ = obj.shape
            if src_h != self.cam_param[0] or src_w != self.cam_param[1] or obj_h != self.cam_param[2] or obj_w != \
                    self.cam_param[3]:
                org_rw, org_rh = self.cam_param[6], self.cam_param[7]
                new_rw, new_rh = float(src_w / obj_w), float(src_h / obj_h)
                org_dx, org_dy = self.cam_param[4], self.cam_param[5]
                new_dx, new_dy = int(new_rw / org_rw * org_dx), int(new_rh / org_rh * org_dy)
                return [src_h, src_w, obj_h, obj_w, new_dx, new_dy, new_rw, new_rh]
            else:
                return self.cam_param

        def bbox_sync(obbox: ndarray, cparam=self.cam_param):
            dx, dy, rw, rh = cparam[4], cparam[5], cparam[6], cparam[7]
            return obbox[0] + dx, obbox[1] + dy, int(obbox[2] * rw), int(obbox[3] * rh)

        def similarity(src_w, src_h, obj_w, obj_h):
            res_w = (obj_w / src_w) if (src_w >= obj_w) else (src_w / obj_w)
            res_h = (obj_h / src_h) if (src_h >= obj_h) else (src_h / obj_h)
            return res_w, res_h

        self.cam_param = check_param(self.main_img, self.sub_img)
        res_boxes = []
        for sbbox, lb in zip(self.main_boxes, self.labels):
            if not lb in self.picks:
                continue
            for obbox in self.sub_boxes:
                srx, sry, srw, srh = sbbox[0], sbbox[1], sbbox[2], sbbox[3]
                obx, oby, obw, obh = bbox_sync(obbox)
                dist_x, dist_y = abs(srx - obx), abs(sry - oby)
                sim_w, sim_h = similarity(srw, srh, obw, obh)
                if (dist_x and dist_y <= self.dist_offset) and (sim_w and sim_h <= self.size_offset):
                    res_x, res_y = (srx + obx) / 2, (sry + oby) / 2
                    res_w, res_h = (srw + obw) / 2, (srh + obh) / 2
                    res_xywh = torch.Tensor([res_x, res_y, res_w, res_h])
                    res_xywh = torch.unsqueeze(res_xywh, dim=0)
                    res_xyxy = xywh2xyxy(res_xywh).squeeze()
                    res_boxes.append(res_xyxy.tolist())
        self.main_img, self.main_boxes, self.labels = None, None, None
        self.sub_img, self.sub_boxes = None, None
        return res_boxes


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
