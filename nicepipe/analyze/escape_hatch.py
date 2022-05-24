from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
# import torch
import cv2
from nicepipe.analyze.base import BaseAnalyzer, AnalysisWorker, AnalysisWorkerCfg


@dataclass
class tapeCfg(AnalysisWorkerCfg):
    device: str = "cuda:0"
    model_class: str = "yolov5n6"
    confidence: float = 0.5  # 0.25
    iou: float = 0.45
    class_filter: Optional[list[int]] = field(default_factory=lambda: [0])
    scale_h: int = 640
    max_fps: int = 30


@dataclass
class DuctTapeDetector(BaseAnalyzer, tapeCfg):
    def init(self):
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            "test/yolov5n6.torchscript",
            device=self.device,
            _verbose=False,
        )
        model.conf = self.confidence
        model.iou = self.iou
        model.classes = self.class_filter
        model.amp = False
        model.eval()
        self.model = model

    def cleanup(self):
        pass

    def analyze(self, img, **_):
        results = self.model(img[..., ::-1], size=self.scale_h)
        # batch-size of 1 means only 1 result anyways
        return results.xyxyn[0].tolist()

    async def _loop(self):
        with torch.no_grad():
            while True:
                img, extra = self.pipe.recv()
                try:
                    results = (0, self.analyze(img, **extra))
                except Exception as e:
                    results = (1, e)
                self.pipe.send(results)


def visualize_output(buffer_and_data):
    imbuffer, tape_results = buffer_and_data
    h, w = imbuffer.shape[:2]
    for det in tape_results:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(
            imbuffer,
            (int(x1 * w), int(y1 * h)),
            (int(x2 * w), int(y2 * h)),
            (1, 0, 0),
            2,
        )


def create_tape_worker(max_fps=tapeCfg.max_fps, lock_fps=tapeCfg.lock_fps, **kwargs):
    if not kwargs:
        kwargs = tapeCfg()

    return AnalysisWorker(
        analyzer=DuctTapeDetector(**kwargs), max_fps=max_fps, lock_fps=lock_fps
    )


if __name__ == "__main__":
    from timeit import Timer

    im1 = cv2.imread("test/bus.jpg")[..., ::-1]
    im2 = cv2.imread("test/owl_place.webp")[..., ::-1]
    ims = [im1, im2]

    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5m6", device="cuda:0"
    )  # _verbose=False
    model.conf = 0.25  # NMS conf
    model.iou = 0.45  # iou thres
    model.classes = [0]  # ppl only
    model.amp = False  # amp is slightly slower

    # expects RGB, size set what height it will downscale to
    # default input size is (640, -1, 3) (HWC) (aspect ratio preserved)
    # can accept either single images or list of images

    timer = Timer("model(im1, size=640)", globals=globals())

    with torch.no_grad():
        results = model(im1, size=640)
        timer.timeit(10)
        print(f"FPS (100 iters): {100/timer.timeit(100)}")

    crops = results.crop(save=False)

    # results.pandas().xyxy[0]

    print("done")
