from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from nicepipe.analyze.base import BaseAnalyzer, AnalysisWorker, AnalysisWorkerCfg
from nicepipe.analyze.utils import letterbox
from nicepipe.utils.logging import ORIGINAL_CWD

# derived from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# Multiple objects, how?
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
    scaleFactor: float = 1.25
    nlevels: int = 6
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    scoreType: int = cv2.ORB_HARRIS_SCORE
    patchSize: int = 31
    fastThreshold: int = 20


# NOTE: THE UNDERLYING OPENCV IMPLEMENTATION IS MULTI-THREADED
# UNREASONABLE VALUES HERE, COUPLED WITH FEATURE-RICH IMAGES
# WILL LAG THE ENTIRE COMPUTER, WHICH MEANS EVERYTHING WILL LAG


@dataclass
class queryDetCfg(orbCfg):
    # dont need as many features for query images
    nfeatures: int = 1000


@dataclass
class testDetCfg(orbCfg):
    # test images will naturally have more potential features
    # but object in focus will attract more of them so dont need too much
    nfeatures: int = 5000


@dataclass
class kpDetCfg(AnalysisWorkerCfg):
    query_detector: queryDetCfg = field(default_factory=queryDetCfg)
    test_detector: testDetCfg = field(default_factory=testDetCfg)
    img_map: dict[str, str] = field(default_factory=dict)
    """map of image name to path"""
    min_matches: int = 10
    """min keypoint matches to consider a detection"""
    use_flann: bool = False
    """use flann-based matcher, its supposed to be faster than brute force at large number of features but... experimentally its slower despite turning up nfeatures"""
    scale_wh: Optional[Tuple[int, int]] = (480, 480)
    """scale props to this resolution"""
    ratio_thres: float = 0.6
    """threshold for keypoint to be considered a match"""
    use_bg_subtraction: bool = True
    """use background subtraction"""
    debug: bool = False
    """whether to pass data needed for debug"""
    max_fps: int = 30


def calc_features(img, detector, descriptor=None, mask=None, keypoints=None):
    """Calculate (keypoints, descriptors, height, width) given an image.

    Args:
        detector (Any): Keypoint detector to use.
        img (np.ndarray): BGR image in HWC order as uint8.
        mask (np.ndarray): bitmask where 1 indicates region of interests.
        keypoints (_type_, optional): Manually specified keypoints to use. Defaults to None.
    """
    assert len(img.shape) == 2, "Image must be grayscale!"
    kps = keypoints if keypoints else detector.detect(img, mask)
    kps, desc = (
        descriptor.compute(img, kps) if descriptor else detector.compute(img, kps)
    )
    return kps, desc, img.shape[0], img.shape[1]


# TODO: figure out the Homography stuff for calibration reasons
# best explanation of homography method I could find:
# http://amroamroamro.github.io/mexopencv/matlab/cv.findHomography.html
# Given very few outliers (according to test), least squares is best
# In brief:
# Least Squares (0): Use all points. Effective only when few outliers
# cv2.RANSAC & cv2.RHO:
#   - attempts to find best set of inliers, but needs threshold
#   - RHO is more accurate but needs more points than RANSAC
# cv2.LMEDS: like voting, needs at least 50% inliers


def find_object(matches, query_kp, test_kp, method=cv2.RANSAC, inlier_thres=5.0):
    # coordinates in query & test image that match
    query_pts = cv2.KeyPoint_convert(
        query_kp, tuple(m.queryIdx for m in matches)
    ).reshape(-1, 1, 2)
    test_pts = cv2.KeyPoint_convert(
        test_kp, tuple(m.trainIdx for m in matches)
    ).reshape(-1, 1, 2)
    transform, mask = cv2.findHomography(query_pts, test_pts, method, inlier_thres)
    return transform, mask


def filter_matches(pairs, ratio_thres=0.75):
    """Use Lowe's ratio test to filter pairs of matches obtained from knnMatch"""
    # pair is (best, 2nd best), check if best is closer by factor compared to 2nd best
    return [
        p[0]
        for p in pairs
        if len(p) == 1 or (len(p) == 2 and p[0].distance < ratio_thres * p[1].distance)
    ]


# https://docs.opencv.org/4.x/de/db2/classcv_1_1KeyPointsFilter.html
# might be alternative method for masking (masking keypoints vs image)


@dataclass
class KPDetector(BaseAnalyzer, kpDetCfg):
    def init(self):
        assert self.img_map, "No query images specified!"

        # All detectors & descriptors available in OpenCV:
        # https://docs.opencv.org/4.x/d5/d51/group__features2d__main.html
        # https://docs.opencv.org/4.x/d3/df6/namespacecv_1_1xfeatures2d.html
        # remember to switch BFMatcher & Flannmatcher code for vector vs binary string based
        self.detector = cv2.ORB_create(**self.test_detector)
        query_detector = cv2.ORB_create(**self.query_detector)
        self.descriptor = cv2.xfeatures2d.BEBLID_create(1.0)

        self.bg_subtractor = (
            cv2.bgsegm.createBackgroundSubtractorGSOC()
            if self.use_bg_subtraction
            else None
        )

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

            # binary descriptors such as ORB, BRISK, BEBLID etc
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
            path = str(ORIGINAL_CWD / path)
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert not im is None, f"Failed to read image {name} from {path}"
            if self.scale_wh:
                im = letterbox(im, self.scale_wh, color=(0, 0, 0))[0]
            self.features[name] = calc_features(im, query_detector, self.descriptor)

    def cleanup(self):
        pass

    def analyze(self, img, **_):
        # TODO: write keypoint tracker
        # https://docs.opencv.org/4.x/d5/dec/classcv_1_1videostab_1_1KeypointBasedMotionEstimator.html
        # will reduce lag if no need to match every frame
        # meanshift will also help
        # worst case... one process per query image?
        mask = self.bg_subtractor.apply(img) if self.use_bg_subtraction else None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t_kp, t_desc, _, _ = calc_features(img, self.detector, self.descriptor, mask)

        o = {}
        results = []
        o["dets"] = results
        if len(t_kp) == 0:
            return o

        # normalize box coords
        wh = img.shape[1::-1]

        matched_kp = []
        for name, (q_kp, q_desc, qh, qw) in self.features.items():
            pairs = self.matcher.knnMatch(
                q_desc, t_desc, k=2
            )  # pairs of (best, 2nd best) matches
            matches = filter_matches(pairs, ratio_thres=self.ratio_thres)
            if len(matches) < self.min_matches:
                continue
            transform, _ = find_object(matches, q_kp, t_kp)
            # tl, bl, br, tr
            box = np.float32(
                ((0, 0), (0, qh - 1), (qw - 1, qh - 1), (qw - 1, 0))
            ).reshape(-1, 1, 2)
            try:
                box = cv2.perspectiveTransform(box, transform).reshape(-1, 2)
                box /= wh
                results.append((name, box))
                if self.debug:
                    matched_kp.append(
                        (
                            name,
                            np.array(
                                tuple(t_kp[m.trainIdx].pt for m in matches)
                            ).reshape(-1, 2)
                            / wh,
                        )
                    )

            except:
                pass
        if self.debug:
            o["debug"] = {
                "all_kp": cv2.KeyPoint_convert(t_kp) / wh,
                "matched_kp": matched_kp,
            }
        return o


def format_output(out, **_):
    # NOTE: box is tl, bl, br, tr
    return [(name, box.tolist()) for name, box in out["dets"]]


def draw_keypoints(im, kps, color=(1, 0, 0), size=1):
    kps = kps.astype(np.uint16)
    a = np.arange(size)
    m = np.array(np.meshgrid(a, a)).T.reshape(-1, 1, 2)
    kps = (np.tile(kps, (size**2, 1, 1)) + m).reshape(-1, 2)
    x_ind = (kps[:, 0] - size // 2).clip(0, im.shape[1] - 1)
    y_ind = (kps[:, 1] - size // 2).clip(0, im.shape[0] - 1)
    im[y_ind, x_ind] = color


def visualize_output(buffer_and_data):
    imbuffer, kp_results = buffer_and_data
    wh = imbuffer.shape[1::-1]

    for (name, box) in kp_results["dets"]:
        box = box * wh
        centre = box.mean(0)
        cv2.polylines(imbuffer, [box.astype(np.int32)], True, (0, 0, 1), 1)
        cv2.putText(
            imbuffer,
            name,
            centre.astype(np.uint16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 1),
        )
    if "debug" in kp_results:
        debug = kp_results["debug"]
        all_kp = debug["all_kp"]
        draw_keypoints(imbuffer, (all_kp * wh).astype(np.uint16), (1, 0, 0))
        match_kps = debug["matched_kp"]
        for (name, kp) in match_kps:
            kp = kp * wh
            draw_keypoints(imbuffer, kp.astype(np.uint16), (0, 1, 0))
            centre = kp.mean(0)
            cv2.putText(
                imbuffer,
                str(kp.shape[0]),
                centre.astype(np.uint16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 1, 0),
            )


def create_kp_worker(
    max_fps=kpDetCfg.max_fps,
    lock_fps=kpDetCfg.lock_fps,
    **kwargs,
):
    if not kwargs:  # empty dicts are false. Okay python.
        kwargs = kpDetCfg()

    return AnalysisWorker(
        analyzer=KPDetector(**kwargs),
        format_output=format_output,
        visualize_output=visualize_output,
        max_fps=max_fps,
        lock_fps=lock_fps,
    )


if __name__ == "__main__":
    from timeit import Timer

    detector = KPDetector(img_map={"owl": "test/owl_query.webp"})
    detector.init()
    test_im = cv2.imread("test/owl_place.webp")

    timer = Timer("detector(test_im)", globals=globals())
    print(f"ms: {timer.timeit(100)/100*1000}")

    results = detector(test_im)
    for (name, rect) in results["dets"]:
        preview = cv2.polylines(test_im, [np.int32(rect)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow("kp_test", preview)
    # cv2.destroyAllWindows()
