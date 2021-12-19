import sys
import os
import torch
import numpy as np
import cv2

from model import (preprocess, detect, box_label,
    get_busway_box_from_prediction)
from time import time
from pathlib import Path
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

print(f'ROOT in {ROOT}')
WEIGHT = ROOT / 'weights'

weight = WEIGHT / 'best-yolov5m-400epochs.pt'

random_state = np.random.RandomState(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}')
model = attempt_load(weight, map_location=device)

IMAGES = ROOT / 'dataset/preprocessed_512x512'
idx = '%03d.jpg' % random_state.randint(0,62)
main_img = cv2.imread(str(IMAGES / idx))

preprocessed = preprocess(main_img,device)
start = time()
pred = detect(model,preprocessed)
print(f'inferenced in {(time() - start)*1E3:.2f}ms')
# labeled_img = box_label(pred,main_img,labels)
# plt.imshow(labeled_img)

lane_box = get_busway_box_from_prediction(pred)
x,y = lane_box.exterior.xy

points = np.int32([x,y])
points = points.transpose()
print(f'amount of convex hull points: {len(points)}')
if points is None:
  print("Cant make polygon from the prediction")
# else:
#   _img = main_img.copy()
#   cv2.fillPoly(_img, [points], (255,0,0))
#   alpha = 0.4
#   image_new = cv2.addWeighted(_img, alpha, main_img, 1 - alpha, 0)
#   plt.imshow(image_new)
#   plt.show()
