from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from onnxruntime import InferenceSession
from pathlib import Path

import nicepipe.models
from nicepipe.predict.base import BasePredictor, PredictionWorker

# ensure images fit input shape and stride requirements
# btw img size must be multiple of stride in both directions
# for example:
# # imgs is array of cv2 imread HWC BGR images
# batch = np.stack([letterbox(x)[0] for x in imgs], 0)
# batch[..., ::-1].transpose(0, 3, 1, 2) # NCHW, RGB
# (dw, dh) is size of padding (one side-only so need x2)
# ratio (also w, h) is resizing ratio
# above 2 are needed to reconstruct the original image
# they are hence useless
# taken from https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py
def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


# modified from https://github.com/ultralytics/yolov5/blob/master/utils/general.py
# replaced all torch operations with numpy equivalents & removed deadcode
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    max_det=300,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates
    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    output = [np.zeros((0, 6))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # NOTE: i removed the "multi-label" features
        # allows predicting the same bbox location if cls conf > thres
        # dataset has same bbox location but different labels
        # not really my vision of "multi-label"...
        # Detections matrix nx6 (xyxy, conf, cls)
        # print(x[:, 5:].max(1, keepdims=True))
        conf = x[:, 5:].max(1, keepdims=True)
        j = x[:, 5:].argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(np.float32)), 1)[
            conf.reshape(-1) > conf_thres
        ]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output


# disable CUDA if CUDA isnt imported
# need better method of detecting cuda if cuda is installed to system rather than using your hack
# meta = session.get_modelmeta().custom_metadata_map
# most impt is meta['stride'] (for letterboxing correct input shape) and meta['names'] (label map)

# forward
def forward(self, im):
    # im is numpy bchw
    y = self.session.run(
        [self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im}
    )[0]
    # y is crazy (n, concat, 85) array
    # can feed to the NMS


@dataclass
class YoloV5Predictor(BasePredictor):
    def init(self):
        self.session = InferenceSession(
            str(Path(nicepipe.models.__path__[0]) / "yolov5n6.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.imghw = (640, 640)
        self.stride = int(self.meta["stride"])
        self.label_map = eval(self.meta["names"])

    def cleanup(self):
        pass

    def predict(self, img, **_):
        x = np.stack(
            (letterbox(img, new_shape=self.imghw, stride=self.stride, auto=False)[0],),
            0,
        )  # NHWC, BGR
        x = x[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)  # NCHW, RGB
        y = self.session.run(
            [self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: x}
        )[
            0
        ]  # (N, CONCAT, 85)
        # TODO: use letterbox info to unpad the coords
        img_preds = non_max_suppression(y)  # [N * (D, 6)] XYXY, CONF, CLS
        out = []
        for img in img_preds:
            img[:, (0, 2)] /= self.imghw[1]
            img[:, (1, 3)] /= self.imghw[0]
            preds = img.tolist()
            for p in preds:
                p[5] = self.label_map[int(p[5])]
            out.append(preds)
        return out


def create_yolo_worker(cfg=None, **kwargs):
    return PredictionWorker(
        predictor=YoloV5Predictor(),
        **kwargs,
    )
