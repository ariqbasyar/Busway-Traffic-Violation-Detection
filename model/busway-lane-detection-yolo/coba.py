import sys
import os
import torch.nn as nn
import torch
import numpy as np
import cv2

from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WEIGHT = ROOT / 'weights'

weight = WEIGHT / 'best-yolov5s-300epochs.pt'

random_state = np.random.RandomState(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(weight, map_location=device)

idx = '%03d.jpg' % random_state.randint(0,62)
main_img = cv2.imread(f'../../dataset/preprocessed_512x512/{idx}')

def detect(main_img):

    img = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    img = np.moveaxis(img,-1,0)
    img = torch.from_numpy(img).to(device)
    img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)

    items=[]
    if pred[0] is not None and len(pred):
        for p in pred[0]:
            score = np.round(p[4].cpu().detach().numpy(),2)
            xmin = int(p[0])
            ymin = int(p[1])
            xmax = int(p[2])
            ymax = int(p[3])

            item = {'bbox' : [(xmin,ymin),(xmax,ymax)],
                    'score': score}

            items.append(item)

    return items

lines = detect(main_img)
print(lines)
