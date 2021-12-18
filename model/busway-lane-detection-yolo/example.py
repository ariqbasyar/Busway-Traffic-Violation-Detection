import sys
import os
import torch
import numpy as np
import cv2

from model import preprocess, detect, box_label
from time import time
from pathlib import Path
from models.experimental import attempt_load

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WEIGHT = ROOT / 'weights'

weight = WEIGHT / 'best-yolov5m-400epochs.pt'

random_state = np.random.RandomState(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}')
model = attempt_load(weight, map_location=device)

idx = '%03d.jpg' % random_state.randint(0,62)
main_img = cv2.imread(f'../../dataset/preprocessed_512x512/{idx}')

preprocessed = preprocess(main_img,device)
start = time()
pred, labels = detect(model,preprocessed)
print(f'inferenced in {(time() - start)*1E3:.2f}ms')
# labeled_img = box_label(pred,main_img,labels)
# plt.imshow(labeled_img)
