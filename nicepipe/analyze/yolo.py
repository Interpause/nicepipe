"""
A large portion of below was taken from ultralytics/yolov5
ofc because yolov5 is their model.
My main modifications were removing reliance on torch and
whatever I deemed extraneous
https://github.com/ultralytics/yolov5/blob/master/utils/general.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from onnxruntime import InferenceSession
from pathlib import Path

import nicepipe.models
from nicepipe.analyze.utils import letterbox
from nicepipe.analyze.base import AnalysisWorkerCfg, BaseAnalyzer, AnalysisWorker


def scale_coords(cur_shape, ori_shape, coords, ratio_pad=None):
    # if padding used is known
    if ratio_pad is None:
        gain = min(cur_shape[0] / ori_shape[0], cur_shape[1] / ori_shape[1])
        pad = (cur_shape[1] - ori_shape[1] * gain) / 2, (
            cur_shape[0] - ori_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, ori_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, ori_shape[0])
    return coords


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


@dataclass
class yoloV5Cfg(AnalysisWorkerCfg):
    confidence: float = 0.7  # 0.25
    nms: float = 0.45  # 0.45
    class_include: Optional[list[str]] = field(default_factory=lambda: ["person"])


@dataclass
class YoloV5Detector(BaseAnalyzer, yoloV5Cfg):
    model_path: str = str(Path(nicepipe.models.__path__[0]) / "yolov5n6.onnx")
    onnx_providers: list[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    imghw: tuple[int, int] = (640, 640)

    def init(self):
        self.session = InferenceSession(self.model_path, providers=self.onnx_providers)
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.stride = int(self.meta["stride"])
        self.label_map = eval(self.meta["names"])

        self._classes = [
            self.label_map.index(c) for c in self.class_include if c in self.label_map
        ]

    def cleanup(self):
        pass

    def _forward(self, img):
        x = np.stack(
            (letterbox(img, new_shape=self.imghw, stride=self.stride, auto=False)[0],),
            0,
        )  # NHWC, BGR, float32
        x = (x / 255)[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)  # NCHW, RGB

        y = self.session.run(
            [self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: x}
        )[
            0
        ]  # output #0: (N, CONCAT, 85)

        dets = non_max_suppression(y, self.confidence, self.nms, self._classes)[
            0
        ]  # [N * (D, 6)] XYXY, CONF, CLS; get only 0th image given batchsize=1
        dets[:, :4] = scale_coords(self.imghw, img.shape, dets[:, :4])
        return dets

    def analyze(self, img, **_):
        if 0 in img.shape:
            return []
        dets = self._forward(img)
        dets[:, (0, 2)] /= img.shape[1]
        dets[:, (1, 3)] /= img.shape[0]
        return [
            (x1, y1, x2, y2, conf, self.label_map[int(cls)])
            for x1, y1, x2, y2, conf, cls in dets.tolist()
        ]


def visualize_outputs(buffer_and_data):
    imbuffer, results = buffer_and_data
    h, w = imbuffer.shape[:2]
    # print(results)
    for det in results:
        x1, y1, x2, y2, conf, cls = det
        cv2.putText(
            imbuffer,
            f"{cls} {conf*100:.1f}%",
            (int(x1 * w) + 2, int(y1 * h) + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (1, 0, 0),
        )
        cv2.rectangle(
            imbuffer,
            (int(x1 * w), int(y1 * h)),
            (int(x2 * w), int(y2 * h)),
            (1, 0, 0),
            1,
        )


def create_yolo_worker(
    max_fps=yoloV5Cfg.max_fps,
    lock_fps=yoloV5Cfg.lock_fps,
    do_profiling=yoloV5Cfg.do_profiling,
    **kwargs,
):
    return AnalysisWorker(
        analyzer=YoloV5Detector(**kwargs),
        visualize_output=visualize_outputs,
        max_fps=max_fps,
        lock_fps=lock_fps,
        do_profiling=do_profiling,
    )
