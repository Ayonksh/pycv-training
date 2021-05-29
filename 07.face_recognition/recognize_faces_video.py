# USAGE
# python recognize_faces_video.py [-v oriImgs/lunch_scene.mp4]

import cv2
import time
import pickle
import imutils
import argparse
import face_recognition
from imutils.video import VideoStream

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to input video")
ap.add_argument("-d", "--detection-method", type = str, default = "hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open("./encodings/encodings.pickle", "rb").read())

if not args.get("video", False):
    print("[INFO] starting video stream...")
    stream = VideoStream(src = 0).start()
    time.sleep(1.0)

else:
    print("[INFO] processing video...")
    stream = cv2.VideoCapture(args["video"])


while True:
    frame = stream.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width = 750)    # 转成750px加快处理速度
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model = args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            # 找到所有匹配到的脸的下标，然后初始化一个字典，对每张脸的匹配数进行计数
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key = counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('video', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

if not args.get("video", False):
    stream.stop()
else:
    stream.release()
cv2.destroyAllWindows()