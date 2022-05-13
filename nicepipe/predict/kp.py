from __future__ import annotations
import asyncio
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from nicepipe.predict.base import BasePredictor, PredictionWorker, predictionWorkerCfg
from nicepipe.utils.logging import ORIGINAL_CWD

# derived from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# TODO: multiple objects... how?
# Method 1: Cluster keypoints using meanshift, then match each cluster (method found online)
# Method 2: Mask each successful detection using gray, then rerun
# Which is more efficient? (probably method 1...) Which is faster...? (probably method 1 as long as code is written well)
# Test method 2 first (easier), then method 1. measure latency and accuracy
# Method 1 might be esp efficient since we dont have to rerun per each query cause each cluster can be assigned a label
# See:
# https://stackoverflow.com/questions/52425355/how-to-detect-multiple-objects-with-opencv-in-c
# https://stackoverflow.com/questions/42938149/opencv-feature-matching-multiple-objects
# https://stackoverflow.com/questions/17357190/using-surf-descriptors-to-detect-multiple-instances-of-an-object-in-opencv
# ^ ROI Sweep, not as stupid as my naive method (i saw one stackoverflow using my method lmao) (still limited by scale, sounds like YOLO)
# https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
# expert confirmation meanshift is the "correct" approach: https://answers.opencv.org/question/17985/detecting-multiple-instances-of-same-object-with-keypoint-matching-approach/
# TBH if ppl were still researching keypoint detection, there might be damn good methods by now
# methods that can match neural networks with much less data (like a better version of HAAR)
# but no it got abandoned in favour of neural networks

# TODO: background removal/subtraction using static image
# TODO: implement multiple objects... even if we dont have duplicate objects, it provides a performance boost to match cluster-wise


@dataclass
class orbCfg:
    """https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html"""

    nfeatures: int = 500
    """max limit on number of features to be detected"""
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31
    fastThreshold: int = 20


@dataclass
class queryDetCfg(orbCfg):
    # dont need as many features for query images
    nfeatures: int = 200


@dataclass
class testDetCfg(orbCfg):
    # test images will naturally have more potential features
    nfeatures: int = 2000


@dataclass
class kpDetCfg(predictionWorkerCfg):
    query_detector: queryDetCfg = field(default_factory=queryDetCfg)
    test_detector: testDetCfg = field(default_factory=testDetCfg)
    img_map: dict[str, str] = field(default_factory=dict)
    """map of image name to path"""
    min_matches: int = 10
    """min keypoint matches to consider a detection"""
    use_flann: bool = False
    """use flann-based matcher, its supposed to be faster than brute force at large number of features but... experimentally its slower despite turning up nfeatures"""
    scale_wh: Optional[Tuple[int, int]] = None
    """downscaling usually wont make sense"""
    ratio_thres: float = 0.7
    """threshold for keypoint to be considered a match"""
    debug: bool = False
    """whether to pass data needed for debug"""


def calc_features(detector, img, keypoints=None):
    """Calculate (keypoints, descriptors, height, width) given an image.

    Args:
        detector (Any): Keypoint detector to use.
        img (np.ndarray): BGR image in HWC order as uint8.
        keypoints (_type_, optional): Manually specified keypoints to use. Defaults to None.
    """
    assert len(img.shape) == 2, "Image must be grayscale!"
    kps, desc = (
        detector.compute(img, keypoints)
        if keypoints
        else detector.detectAndCompute(img, None)
    )
    return kps, desc, img.shape[0], img.shape[1]


# TODO: figure out the Homography stuff for calibration reasons
def find_object(matches, query_kp, test_kp):
    # coordinates in query & test image that match
    query_pts = cv2.KeyPoint_convert(
        query_kp, tuple(m.queryIdx for m in matches)
    ).reshape(-1, 1, 2)
    test_pts = cv2.KeyPoint_convert(
        test_kp, tuple(m.trainIdx for m in matches)
    ).reshape(-1, 1, 2)
    transform, mask = cv2.findHomography(query_pts, test_pts, cv2.RANSAC, 5.0)
    return transform, mask


def filter_matches(pairs, ratio_thres=0.75):
    """Use Lowe's ratio test to filter pairs of matches obtained from knnMatch"""
    # pair is (best, 2nd best), check if best is closer by factor compared to 2nd best
    return [
        p[0]
        for p in pairs
        if len(p) == 1 or (len(p) == 2 and p[0].distance < ratio_thres * p[1].distance)
    ]


# https://docs.opencv.org/3.4/de/db2/classcv_1_1KeyPointsFilter.html
# might be alternative method for masking (masking keypoints vs image)


@dataclass
class KPDetPredictor(BasePredictor, kpDetCfg):
    def init(self):
        # ORB was used instead of SIFT or others because I cannot find sufficient info
        # and apparently is the most efficient.
        # https://docs.opencv.org/3.4/d5/d51/group__features2d__main.html
        # TODO: consider/try more detectors
        # remember to switch BFMatcher & Flannmatcher code for vector vs binary string based
        self.detector = cv2.ORB_create(**asdict(orbCfg()))

        # NOTE: no documentation exists that i cannot figure out what
        # parameters exist or do... values here are hardcoded from tutorial
        if self.use_flann:
            # checks affect recursion level of flann to increase accuracy
            # ...but make it insanely large or 0 seems to do nothing
            # likely because it is the "upper-limit" & hence only affects
            # if the target is hidden/not present
            search_params = dict(checks=50)

            # SURF, SIFT, etc
            # FLANN_INDEX_KDTREE = 1
            # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

            # ORB, BRISK, etc
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,  # 12
                key_size=12,  # 20
                multi_probe_level=1,  # 2
            )

            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # dont use builtin crossCheck, its significantly slower than ratio test
            self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        self.features = {}

        for name, path in self.img_map.items():
            # TODO: a resolver function
            im = cv2.imread(str(ORIGINAL_CWD / path), cv2.IMREAD_GRAYSCALE)
            assert not im is None, f"Failed to read image {name} from {path}"
            self.features[name] = calc_features(self.detector, im)

    def cleanup(self):
        pass

    def predict(self, img, **_):
        # TODO: write keypoint tracker
        # https://docs.opencv.org/4.x/d5/dec/classcv_1_1videostab_1_1KeypointBasedMotionEstimator.html
        # will reduce lag if no need to match every frame
        # worst case... one process per query image?
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_kp, t_desc, _, _ = calc_features(self.detector, img)

        results = []
        matched_kp = []
        for name, (q_kp, q_desc, qh, qw) in self.features.items():
            pairs = self.matcher.knnMatch(
                q_desc, t_desc, k=2
            )  # pairs of (best, 2nd best) matches
            matches = filter_matches(pairs, ratio_thres=self.ratio_thres)
            if self.debug:
                matched_kp.extend(t_kp[m.trainIdx].pt for m in matches)
            if len(matches) < self.min_matches:
                continue
            transform, _ = find_object(matches, q_kp, t_kp)
            rect = np.float32(
                ((0, 0), (0, qh - 1), (qw - 1, qh - 1), (qw - 1, 0))
            ).reshape(-1, 1, 2)
            rect = cv2.perspectiveTransform(rect, transform)
            results.append((name, rect))
        o = {}
        o["dets"] = results
        # debug code will only work if img isnt rescaled
        if self.debug:
            o["debug"] = {
                "all_kp": cv2.KeyPoint_convert(t_kp).tolist(),
                "matched_kp": matched_kp,
            }
        return o


def create_kp_worker(
    scale_wh=kpDetCfg.scale_wh,
    max_fps=kpDetCfg.max_fps,
    lock_fps=kpDetCfg.lock_fps,
    **kwargs,
):
    if not kwargs:  # empty dicts are false. Okay python.
        kwargs = kpDetCfg()

    async def process_input(img, **extra):
        if scale_wh is None:
            return img, extra
        return await asyncio.to_thread(cv2.resize, img, scale_wh), extra

    async def format_output(out, **_):
        out.pop("debug", None)
        return out

    return PredictionWorker(
        predictor=KPDetPredictor(**kwargs),
        process_input=process_input,
        format_output=format_output,
        max_fps=max_fps,
        lock_fps=lock_fps,
    )


if __name__ == "__main__":
    from timeit import Timer

    predictor = KPDetPredictor(img_map={"owl": "test/owl_query.webp"})
    predictor.init()
    test_im = cv2.imread("test/owl_place.webp")

    timer = Timer("predictor.predict(test_im)", globals=globals())
    print(f"ms: {timer.timeit(100)/100*1000}")

    results = predictor.predict(test_im)
    for (name, rect) in results["dets"]:
        preview = cv2.polylines(test_im, [np.int32(rect)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("kp_test", preview)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
