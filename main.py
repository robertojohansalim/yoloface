from face_detector import YoloDetector
import numpy as np
from PIL import Image


img_path = "./test_img/single_face.png"


print("Loading Model")
model = YoloDetector(target_size=720,min_face=90, device='cpu')
print("Loading Image")
orgimg = np.array(Image.open(img_path))
print("Predicting")
bboxes,points = model.predict(orgimg)
print(bboxes, points)