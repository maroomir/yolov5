# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# TheEye project by maroomir@gmail.com
import argparse
import os.path
import sys
from pathlib import Path

import numpy
import torch
import cv2
import torch.backends.cudnn

from models.experimental import attempt_load
from utils.comments import CommentWriter, RealDistanceToObject
from utils.datasets import LoadFusingImages
from utils.general import print_args, check_img_size, check_imshow, check_requirements, check_suffix, increment_path, \
    non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.fuse import Fusing, get_diff
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Convert the relative path
DEBUG_MODE = False

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to system path
if DEBUG_MODE:
    import os
    import matplotlib
    import matplotlib.pyplot

    matplotlib.use('TKAgg')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@torch.no_grad()
def detect(weights,  # Essential parameter
           sources,  # Essential parameter
           cal_mode,
           cal_path,
           view_img,
           ):
    # Initialize the default parameter
    device = select_device()
    default_size = [640, 640]
    if not cal_mode:
        fuse = Fusing(str(cal_path))
    # Initialize the directory
    set_logging()
    project = ROOT / 'runs/detect'
    name = 'exp'
    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Initialize the source
    src_main = str(sources[0] if isinstance(sources, list) else sources)
    src_sub = str(sources[1] if isinstance(sources, list) else sources)
    # Load model
    weight_main = str(weights[0] if isinstance(weights, list) else weights)
    weight_sub = str(weights[1] if isinstance(weights, list) else weights)
    model_main = torch.jit.load(weight_main) if 'torchscript' in weight_main else attempt_load(weight_main,
                                                                                               map_location=device)
    model_sub = torch.jit.load(weight_sub) if 'torchscript' in weight_sub else attempt_load(weight_sub,
                                                                                            map_location=device)
    stride_main = int(model_main.stride.max())  # model stride
    stride_sub = int(model_sub.stride.max())
    names_main = model_main.module.names if hasattr(model_main, 'module') else model_main.names  # get class names
    names_sub = model_sub.module.names if hasattr(model_sub, 'module') else model_sub.names  # get class names
    main_size = check_img_size(default_size, s=stride_main)
    sub_size = check_img_size(default_size, s=stride_sub)
    # Load Dataset
    dataset = LoadFusingImages(main_path=src_main, sub_path=src_sub,
                               main_size=main_size, sub_size=sub_size,
                               main_stride=stride_main, sub_stride=stride_sub)
    # Run interface
    if device.type != 'cpu':  # run the once only using CUDA
        model_main(torch.zeros(1, 3, *main_size).to(device).type_as(next(model_main.parameters())))
        model_sub(torch.zeros(1, 3, *sub_size).to(device).type_as(next(model_sub.parameters())))
    time_laps, num_seen = [0.0, 0.0, 0.0], 0
    for m_path, m_img, m_org, s_path, s_img, s_org in dataset:
        time1 = time_sync()
        m_img = torch.from_numpy(m_img).to(device)
        s_img = torch.from_numpy(s_img).to(device)
        m_img = m_img.float() / 255.0
        s_img = s_img.float() / 255.0
        # Expand the dimension for batch
        if len(m_img.shape) == 3:
            m_img = m_img[None]
        if len(s_img.shape) == 3:
            s_img = s_img[None]
        time2 = time_sync()
        time_laps[0] += time2 - time1
        # Inference
        pred_main = model_main(m_img, augment=False, visualize=False)[0]
        pred_sub = model_sub(s_img, augment=False, visualize=False)[0]
        time3 = time_sync()
        time_laps[1] += time3 - time2
        # Non Max Suppression
        pred_main = non_max_suppression(pred_main, conf_thres=0.25, iou_thres=0.45, classes=None,
                                        agnostic=False, max_det=1000)
        pred_sub = non_max_suppression(pred_sub, conf_thres=0.25, iou_thres=0.45, classes=None,
                                       agnostic=False, max_det=1000)
        time_laps[2] += time_sync() - time3
        # Prediction process
        for (i, m_det), (j, s_det) in zip(enumerate(pred_main), enumerate(pred_sub)):
            num_seen += 1
            _m_path, m_log, _m_org = m_path, '', m_org.copy()
            _s_path, s_log, _s_org = s_path, '', s_org.copy()
            # convert to path
            _m_path = Path(_m_path)
            _s_path = Path(_s_path)
            m_log += "%gx%g" % m_img.shape[2:]
            s_log += "%gx%g" % s_img.shape[2:]
            _m_gain = torch.tensor(_m_org.shape)[[1, 0, 1, 0]]  # for normalization
            _s_gain = torch.tensor(_s_org.shape)[[1, 0, 1, 0]]
            main_annotator = Annotator(_m_org, line_width=3, example=str(names_main))
            sub_annotator = Annotator(_s_org, line_width=3, example=str(names_sub))

            def write_prediction(annotator: Annotator,
                                 image, origin, detection, names, log):
                xyxy_boxes = []
                labels = []
                if len(detection):
                    # Rescale prediction boundary boxes
                    detection[:, :4] = scale_coords(image.shape[2:], detection[:, :4], origin.shape).round()
                    # Logging
                    for _class in detection[:, -1].unique():
                        num = (detection[:, -1] == _class).sum()
                        log += f" {num} {names[int(_class)]}{'s' * (num > 1)}, "
                    # Write results
                    for *xyxy, _conf, _class in reversed(detection):
                        _class = int(_class)
                        xyxy_boxes += [xyxy]
                        labels += [names[_class]]
                        label = f"{names[_class]} {_conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(_class, True))
                        commenter = CommentWriter(RealDistanceToObject(xyxy, names[_class], _conf))
                        comment = commenter()
                        annotator.box_comment(xyxy, comment, colors(_class, True))
                return annotator, log, xyxy_boxes, labels

            main_annotator, m_log, m_xyxy, m_lbs = \
                write_prediction(main_annotator, m_img, _m_org, m_det, names_main, m_log)
            sub_annotator, s_log, s_xyxy, s_lbs = \
                write_prediction(sub_annotator, s_img, _s_org, s_det, names_sub, s_log)
            # Print time
            print(f"{m_log}Done. ({time3 - time2:.3f}s)", end='')
            print(f"{s_log}Done. ({time3 - time2:.3f}s)")
            if DEBUG_MODE:
                res_main = main_annotator.result()
                res_sub = sub_annotator.result()
                cv2.imshow(str(m_path), res_main)
                cv2.imshow(str(s_path), res_sub)
                cv2.waitKey(1)  # 1 ms
                cv2.waitKey(1)
            else:
                # Update to the fusing
                if cal_mode:
                    get_diff(main_img=_m_org, main_box=m_xyxy[0], sub_img=_s_org, sub_box=s_xyxy[0],
                             save_path=cal_path)
                else:
                    fuse.input_main(im=_m_org, xyxy=m_xyxy, lbs=m_lbs)
                    fuse.input_sub(im=_s_org, xyxy=s_xyxy)
                    fuse_boxes = fuse.do()
                    wait_time = 1  # 1ms
                    if len(fuse_boxes) > 0:
                        for xyxy in fuse_boxes:
                            main_annotator.box_label(xyxy, "fusing", color=(0, 0, 0))
                        wait_time = 1000  # 1s
                    res_main = main_annotator.result()
                    res_sub = sub_annotator.result()
                    if view_img:
                        cv2.imshow(str(m_path), res_main)
                        cv2.imshow(str(s_path), res_sub)
                        cv2.waitKey(wait_time)  # 1 ms or 1s
                        cv2.waitKey(wait_time)

    # Print results
    speed = tuple(time / num_seen * 1E3 for time in time_laps)  # Speed
    print(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *main_size)}" % speed)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--sources', nargs='+', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob')
    parser.add_argument('--cal-mode', default=False, action='store_true', help='calibration mode')
    parser.add_argument('--cal-path', type=str, default=ROOT / 'data/theeye/cal_data.json', help='calibration file')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


if __name__ == "__main__":
    option = parse_opt()
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(option))
