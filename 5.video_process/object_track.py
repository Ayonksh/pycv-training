import cv2
import numpy as np

# 实例化OpenCV's multi-object tracker
trackers = cv2.legacy.MultiTracker_create()
vc = cv2.VideoCapture('./test.avi')

while True:
    ret, frame = vc.read()
    if frame is None:
        break

    # 视频文件太大，把尺寸变小
    (h, w) = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # 追踪结果
    (success, boxes) = trackers.update(frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('video', frame)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('s'):
        # 按s选择一个区域，然后再按空格开始
        box = cv2.selectROI('video', frame, fromCenter = False, showCrosshair = True)

        # 新建一个追踪器
        tracker = cv2.legacy.TrackerKCF_create()
        trackers.add(tracker, frame, box)

    elif key == 27:
        break

vc.release()
cv2.destroyAllWindows()