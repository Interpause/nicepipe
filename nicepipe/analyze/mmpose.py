"""
Below is a replica of the pre & post-processing pipeline of 
https://github.com/open-mmlab/mmpose/blob/master/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py

I took liberties when implementing it for conciseness & performance
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from colorsys import hsv_to_rgb
from pathlib import Path
import random

import cv2

# import cython
import numpy as np

# https://github.com/tryolabs/norfair/tree/master/docs
from norfair import Detection, Tracker
from norfair.tracker import TrackedObject

from onnxruntime import InferenceSession

from nicepipe.analyze.base import AnalysisWorker

import nicepipe.models
from nicepipe.analyze.yolo import YoloV5Detector, yoloV5Cfg

# import last to avoid compiling others, which could cause issues
# import nicepipe.utils.cython_hack

# the feels when 50% of the lag is from normalizing the image
# should normalize the crops instead i guess
# TODO: figure out to how optimize or parallelize the taylor (70%) and guassian (20%) parts
# seriously post-processing shouldnt take more time than the model inference...

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

# fmt: off
coco2mp_map = (
    0, 65, 1, 68, 62, 2, 59, 3, 4, 77,
    71, 5, 6,  7,  8, 9, 10, 110, 131,
    98, 119,  94, 115, 11, 12, 13, 14,
    15, 16, 19, 22, 17, 20,
)
# fmt: on


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


# yanked straight from mmpose. my brain too puny to vectorize this.
# @cython.compile
def taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1]
            - heatmap[py - 1][px + 1]
            - heatmap[py + 1][px - 1]
            + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px]
        )
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


# yanked straight from mmpose, vectorizing this is too hard for my puny brain esp cause cv2 is used
# @cython.compile
def gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


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

    preds = np.tile(ind, (1, 1, 2)).astype(np.float32)
    preds[..., 0] = preds[..., 0] % w
    preds[..., 1] = preds[..., 1] // w
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)

    heatmaps = np.log(np.maximum(gaussian_blur(heatmaps, 11), 1e-10))
    for i in range(n):
        for j in range(k):
            preds[i][j] = taylor(heatmaps[i][j], preds[i][j])

    # mask = 1 < preds[:, :, 0] < w - 1 & 1 < preds[:, :, 1] < h - 1
    # diff = np.array(tuple(heatmaps))

    return remap_keypoints(preds, centers, scales, (w, h)), maxvals


def create_pose_distance_calculator(dist_threshold=1 / 40, conf_threshold=0.5):
    """gauge pose distance by number of keypoints under a threshold of euclidean distance"""

    # TODO: possible to swap out distance functions? like using manhattan or cosine?
    # btw if you wanted to insert image feature vector association (like deepSORT), it would be here.
    def pose_distance(pose: Detection, pose_track: TrackedObject):
        dists = np.linalg.norm(pose.points - pose_track.estimate, axis=1)
        num_match = np.count_nonzero(
            (dists < dist_threshold)
            * (pose.scores > conf_threshold)
            * (pose_track.last_detection.scores > conf_threshold)
        )
        return 1 / max(num_match, 1)

    return pose_distance


@dataclass
class mmposeCfg(yoloV5Cfg):
    crop_pad: float = 1.25
    keypoints_include: Optional[list[Any]] = field(default_factory=lambda: [(0, 133)])
    """list of tuples or int indexes. tuples are converted to slices to index. there are 133 keypoints in coco wholebody."""


# NOTE: declaring properties on mmposeCfg = public settings
# declaring properties on MMPoseDetector = secret settings. Neat.


@dataclass
class MMPoseDetector(YoloV5Detector, mmposeCfg):
    pose_model_path: str = str(Path(nicepipe.models.__path__[0]) / "vipnas_res50.onnx")
    input_wh: tuple[int, int] = (192, 256)
    crop_pad: float = 1.25
    # idk are these imagenet's standardization values?
    mean_rgb: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: tuple[float, float, float] = (0.229, 0.224, 0.225)

    kp_dist_thres: float = 1 / 40
    """normalized distance for 2 keypoints to be considered tracked"""
    kp_conf_thres: float = 0.4
    """min confidence to consider a keypoint for tracking"""
    tracked_kps: list[int] = field(default_factory=lambda: [0, 11, 12, 23, 24])
    """which keypoints to use for tracking"""
    dist_thres: float = (
        0.8  # e.g. 1/2 means tracking the equivalent of 2 points perfectly
    )
    """overall distance threshold, above which is considered a separate object"""

    def init(self):
        super().init()
        self.pose_session = InferenceSession(
            self.pose_model_path, providers=self.onnx_providers
        )
        self._ratio_wh = self.input_wh[0] / self.input_wh[1]
        if self.keypoints_include is None:
            self.keypoints_include = coco2mp_map
        self._include_key = np.r_[
            tuple(
                i if isinstance(i, int) else slice(*i) for i in self.keypoints_include
            )
        ]

        # https://github.com/tryolabs/norfair/tree/master/docs#arguments
        self._tracker = Tracker(
            distance_function=create_pose_distance_calculator(
                self.kp_dist_thres, self.kp_conf_thres
            ),
            distance_threshold=self.dist_thres,  # see pose_distance_calculator for how it is calculated
            detection_threshold=self.kp_conf_thres,
            # "HP" +1 whenever detected, -1 whenever missed
            hit_inertia_max=5,  # "max HP" of a tracked object
            hit_inertia_min=1,  # "starting HP" of a tracked object
            initialization_delay=3,  # frames before tracked object is "spawned"
            past_detections_length=0,
        )

    def cleanup(self):
        super().cleanup()

    def _forward(self, img):
        if 0 in img.shape:
            return np.zeros((0, len(self._include_key), 4))
        dets = super()._forward(img)
        if len(dets) == 0:
            return np.zeros((0, len(self._include_key), 4))

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
        return out  # (n, kp, 4) 4 = (x, y, conf, id)

    def analyze(self, img, **_):
        dets = [
            Detection(
                points=d[self.tracked_kps, :2],
                scores=d[self.tracked_kps, 2],
                data=d,
                label=0,
            )
            for d in self._forward(img)
        ]
        # norfair tracker can account for period
        # TODO: pass fps info into here, somehow
        tracks = self._tracker.update(detections=dets)
        return {t.id: t.last_detection.data for t in tracks}


# coco keypoints start from 1 rather than 0
# but their location inside the array is ofc from 0
# hence all coco ids are decremented by 1 below

# TODO: output_formatter for the plain results
def coco_wholebody2mp_pose(kp):
    # drop precision to compress
    return (int(kp[0] * 255), int(kp[1] * 255), 0, int(kp[2] * 255))


def format_as_mp_pose(out, **_):
    # user has manually selected coco_keypoints via keypoints_include
    if len(next(iter(out.values()), [])) == 33:
        return {
            i: tuple(coco_wholebody2mp_pose(kp) for kp in pose)
            for i, pose in out.items()
        }
    return {
        i: tuple(coco_wholebody2mp_pose(pose[n]) for n in coco2mp_map)
        for i, pose in out.items()
    }


id_color_map = {}


def visualize_outputs(buffer_and_data, confidence_thres=0.5):
    imbuffer, results = buffer_and_data
    for id, pose in results.items():
        c = id_color_map.get(id, None)
        if c is None:
            c = id_color_map[id] = random.random()

        color = hsv_to_rgb(c / len(results), 1, 1)
        for (x, y, c, i) in pose:
            if i == 1:
                cv2.putText(
                    imbuffer,
                    f"#{id}",
                    (int(x * imbuffer.shape[1]), int(y * imbuffer.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                )
                continue
            if c < confidence_thres:
                continue
            # print(f"Point {i} at {int(x*imbuffer.shape[1])},{int(y*imbuffer.shape[0])}")
            cv2.circle(
                imbuffer,
                (int(x * imbuffer.shape[1]), int(y * imbuffer.shape[0])),
                1,
                color,
                cv2.FILLED,
            )


def create_mmpose_worker(
    max_fps=mmposeCfg.max_fps,
    lock_fps=mmposeCfg.lock_fps,
    do_profiling=mmposeCfg.do_profiling,
    **kwargs,
):
    return AnalysisWorker(
        analyzer=MMPoseDetector(**kwargs),
        visualize_output=visualize_outputs,
        format_output=format_as_mp_pose,
        max_fps=max_fps,
        lock_fps=lock_fps,
        do_profiling=do_profiling,
    )
