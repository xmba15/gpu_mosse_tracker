#!/usr/bin/env python
import argparse
import os
import sys
import time

import cv2
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
if True:  # noqa F402
    from mosse_tracker import MosseTracker


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    cap = cv2.VideoCapture(args.video_path)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ok, frame = cap.read()
    if not ok:
        print(f"cannot read: {args.video_path}")
        sys.exit()

    tracker = MosseTracker(img_shape=shape, use_gpu=args.use_gpu)

    # bbox = cv2.selectROI(frame, False)
    bbox = (210, 425, 122, 80)

    if not tracker.init(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bbox):
        print("failed to initialize tracking")
        sys.exit()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_bbox = tracker.update(frame_gray)

        if not output_bbox:
            print("failed to track")
            break

        x_min, y_min, w, h = output_bbox
        cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), (255, 0, 0), 2, 1)
        cv2.imshow("tracked result", frame)

        k = cv2.waitKey(int(1000 / fps)) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
