"""
Below is a replica of the pre & post-processing pipeline of 
https://github.com/open-mmlab/mmpose/blob/master/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py

I took liberties when implementing it for conciseness & performance
"""

from __future__ import annotations
from dataclasses import dataclass, field
from colorsys import hsv_to_rgb
from typing import Any, Optional

import cv2
import numpy as np

from onnxruntime import InferenceSession
from pathlib import Path

from nicepipe.analyze.base import AnalysisWorker

import nicepipe.models
from nicepipe.analyze.yolo import YoloV5Detector, yoloV5Cfg

# the feels when 50% of the lag is from normalizing the image
# should normalize the crops instead i guess
# import pprofile
# profiler = pprofile.Profile()

# Pipeline steps
# 1. convert image from BGR to RGB
# 2. For each bbox (abs_XYWH) calculate abs centre (x,y) & std scale (sx,sy)
#   a. padding is 1.25 by default for coco wholebody
#   b. (sx,sy) is divided by 200 for some inter-dataset compatability reason
# 3. Do some weird affine transformation cropping shit. Screw that tho
#   a. Crop out image with padding using centrescale as (HWC)
#   b. at this stage, image is still uint8 0-255 RGB HWC
#   c. image isnt resized, only cropped. If bbox exceeds image, pad with black


# ann_info:
#   image_size: model input size/bbox size [192, 256] (w, h)
#   num_joints: 133 coco whole-body

# TODO: should profile this line by line, esp the numpy operations


def bbox_xyxy2cs(bbox: np.ndarray, ratio_wh=192 / 256, pad=1.25):
    """Converts abs bbox N*(x,y,x,y) to abs centre N*(x,y) & abs scale N*(sx,sy)"""
    x1, y1, x2, y2 = bbox.astype(np.float32).T[:4]
    w, h = x2 - x1, y2 - y1
    centers = np.stack((x1 + 0.5 * w, y1 + 0.5 * h), axis=1)

    mask = w > ratio_wh * h
    h[mask] = w[mask] / ratio_wh
    w[~mask] = h[~mask] * ratio_wh
    scales = np.stack((w, h), axis=1) * pad

    return centers, scales


def crop_bbox(
    img: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    crop_wh: tuple[int, int] = (192, 256),
):
    """From 1 HWC img, crop N*HWC crops from N*(x,y) & N*(sx,sy)"""
    im_height, im_width = img.shape[:2]
    dw, dh = crop_wh
    N = centers.shape[0]
    crops = np.zeros((N, dh, dw, 3), dtype=img.dtype)

    # source xyxy
    s = scales / 2
    rects = np.tile(centers, 2) + np.concatenate((-s, s), axis=1)

    # calculate margin required when rect exceeds image
    ml = np.maximum(-rects[:, 0], 0).astype(int)
    mt = np.maximum(-rects[:, 1], 0).astype(int)
    # mr = np.maximum(rects[:, 2] - im_width, 0)
    # mb = np.maximum(rects[:, 3] - im_height, 0)

    # clip rects to within image
    rects[:, (0, 2)] = rects[:, (0, 2)].clip(0, im_width)
    rects[:, (1, 3)] = rects[:, (1, 3)].clip(0, im_height)
    rects = rects.astype(int)
    scales = np.ceil(scales).astype(int)

    for n, ((x1, y1, x2, y2), (w, h), l, t) in enumerate(zip(rects, scales, ml, mt)):
        roi = img[y1:y2, x1:x2, :]
        crop = np.zeros((h, w, 3), dtype=img.dtype)
        crop[t : t + y2 - y1, l : l + x2 - x1] = roi
        crops[n, ...] = cv2.resize(crop, (dw, dh))

    return crops


def remap_keypoints(
    coords: np.ndarray,  # (n, keypoints, 2)
    center: np.ndarray,  # (n, 2)
    scale: np.ndarray,  # (n, 2)
    heatmap_wh: tuple[int, int],
):
    factor = (scale / heatmap_wh).reshape(-1, 1, 2)
    center = center.reshape(-1, 1, 2)
    scale = scale.reshape(-1, 1, 2)
    return coords * factor + center - scale * 0.5


def heatmap2keypoints(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray):
    n, k, h, w = heatmaps.shape

    # processing heatmap into coords and conf
    tmp1 = heatmaps.reshape((n, k, -1))
    ind = np.argmax(tmp1, 2).reshape((n, k, 1))
    maxvals = np.amax(tmp1, 2).reshape((n, k, 1))

    # TODO: there is a good reason they do post-processing
    # your coordinates look more like grid points without it
    # using the heatmap to calculate "inbetweens" is apparently
    # a post-processing step

    preds = np.tile(ind, (1, 1, 2)).astype(np.float32)
    preds[..., 0] = preds[..., 0] % w
    preds[..., 1] = preds[..., 1] // w
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    preds = remap_keypoints(preds, centers, scales, (w, h))

    return preds, maxvals


@dataclass
class mmposeCfg(yoloV5Cfg):
    crop_pad: float = 1.25
    keypoints_include: Optional[list[Any]] = field(default_factory=lambda: [(17,)])
    """list of tuples or int indexes. tuples are converted to slices to index. there are 133 keypoints in coco wholebody."""


@dataclass
class MMPoseDetector(YoloV5Detector, mmposeCfg):
    pose_model_path: str = str(Path(nicepipe.models.__path__[0]) / "vipnas_res50.onnx")
    input_wh: tuple[int, int] = (192, 256)
    crop_pad: float = 1.25
    # idk are these imagenet's standardization values?
    mean_rgb: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def init(self):
        super().init()
        self.pose_session = InferenceSession(
            self.pose_model_path, providers=self.onnx_providers
        )
        self._ratio_wh = self.input_wh[0] / self.input_wh[1]
        if self.keypoints_include is None:
            self.keypoints_include = slice()
        self._include_key = np.r_[
            tuple(
                i if isinstance(i, int) else slice(*i) for i in self.keypoints_include
            )
        ]

    def cleanup(self):
        super().cleanup()
        # profiler.dump_stats("profile-mmpose.lprof")

    def analyze(self, img, **_):
        # with profiler:
        if 0 in img.shape:
            return []
        dets = self._forward(img)
        if len(dets) == 0:
            return []

        c, s = bbox_xyxy2cs(dets[:, :4], ratio_wh=self._ratio_wh, pad=self.crop_pad)
        crops = crop_bbox(img, c, s, self.input_wh)  # crop on original image not scaled
        crops = (crops[..., ::-1] / 255 - self.mean_rgb) / self.std_rgb
        x = crops.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC to NCHW

        # (N, keypoints (133 coco wholebody), 64, 48) (64, 48) is heatmap
        # get output #0
        y = self.pose_session.run(
            [self.pose_session.get_outputs()[0].name],
            {self.pose_session.get_inputs()[0].name: x},
        )[0][:, self._include_key, ...]

        # given same input, ~0 distance from mmpose's version when post_process=None
        # aka its correctly implemented
        coords, conf = heatmap2keypoints(y, c, s)
        # normalize coords
        ncoords = coords / img.shape[1::-1]
        # coco wholebody ids
        ids = (np.arange(conf.size) + 1).reshape(conf.shape)
        out = np.concatenate((ncoords, conf, ids), axis=2)

        return out  # (n, kp, (x,y,conf,id))


def visualize_outputs(buffer_and_data):
    imbuffer, results = buffer_and_data
    for n, pose in enumerate(results):
        color = hsv_to_rgb(n / len(results), 1, 1)
        for (x, y, _, i) in pose:
            # print(f"Point {i} at {int(x*imbuffer.shape[1])},{int(y*imbuffer.shape[0])}")
            cv2.circle(
                imbuffer,
                (int(x * imbuffer.shape[1]), int(y * imbuffer.shape[0])),
                2,
                color,
            )


def create_mmpose_worker(
    max_fps=mmposeCfg.max_fps, lock_fps=mmposeCfg.lock_fps, **kwargs
):
    return AnalysisWorker(
        analyzer=MMPoseDetector(**kwargs),
        visualize_output=visualize_outputs,
        max_fps=max_fps,
        lock_fps=lock_fps,
    )
