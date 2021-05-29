import os
import cv2
import pickle
import face_recognition
from imutils import paths

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("./dataset"))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    # 从图片路径提取人名
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    # 把图片从 OpenCV 的 BGR 顺序转换成 dlib 的 RGB 顺序
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model = "cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()