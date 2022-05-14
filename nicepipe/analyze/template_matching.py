from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np
from .base import BaseAnalyzer


def scale_image(image, target_wh):
    """specify -1 for a dimension to ignore it"""
    h, w = image.shape[:2]
    tw, th = target_wh
    # scale factor in both x & y axis
    rx, ry = None, None
    if th > 0:
        ry = th / float(h)
    if tw > 0:
        rx = tw / float(w)
    # final chosen scaling factor
    r = 1
    if rx is None:
        r = ry
    elif ry is None:
        r = rx
    else:
        # if shrink (scale < 1), min will be the greatest shrink, hence fitting the target_wh
        # if grow (scale > 1), min will be the smallest growth, hence fitting the target_wh
        r = min(rx, ry)
    inter = cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, None, r, r, interpolation=inter)


@dataclass
class TemplateMatchingAnalyzer(BaseAnalyzer):
    im_map: dict[str, str] = field(default_factory={})
    """map of image name to path"""
    grayscale: bool = True
    """whether to use grayscale image for matching"""
    canny: bool = True
    """whether to use edges for matching"""
    img_wh: tuple[int, int] = (1280, 720)
    """input image size"""
    sizes: list[float] = field(default_factory=lambda: list(range(1.5, 4.0, 0.5)))
    """factors to shrink templates by relative to img_wh during matching"""
    canny_thres: tuple[int, int] = (50, 200)

    def init(self):
        # each image template at different sizes relative to img_wh
        self.templates: tuple[dict[str, np.ndarray]] = tuple({} for _ in self.sizes)

        self.img_wh = np.array(self.img_wh)

        for name, path in self.im_map.items():
            # TODO: test grayscale (w/o canny) vs color
            # i got a suspicion the edge method wont work for our case
            # might have to go back to keypoints if this doesnt work

            src = cv2.imread(path)
            # convert bgr to gray instead of reading grayscale
            # cause there is (somehow) a disrepency between reading and converting
            # and the input images will almost certainly be converted rather than read
            if self.grayscale:
                src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            for i, size in enumerate(self.sizes):
                # should scaling be before or after canny?
                im = scale_image(src, self.img_wh / size)
                # NOTE: either make this adjustable per image or hardcode it.
                if self.grayscale and self.canny:
                    im = cv2.Canny(im, *self.canny_thres, L2gradient=True)
                self.templates[i][name] = im

    def cleanup(self):
        pass

    def analyze(self, img, **_):
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.canny:
                img = cv2.Canny(img, *self.canny_thres, L2gradient=True)

        # [(name, x, y, w, h)]
        results = []
        for size_set in self.templates:
            for name, template in size_set.items():
                heatmap = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
                _, v, _, loc = cv2.minMaxLoc(heatmap)
                if v > self.match_thres:  # TODO: this is a placeholder threshold value
                    results.append((name, *loc, template.shape[1], template.shape[0]))
        return results


# https://pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/
# see above how to properly handle multi-object template matching
# taking the best match at each scale is clearly incorrect
# for that approach its supposed to just take the best match at any scale
# then maybe threshold
# possibly threshold at every scale or smth... need some live calibration shit tho

if __name__ == "__main__":
    pass
