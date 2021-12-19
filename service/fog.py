import sys
import cv2
import socket
import time
import os
import torch

from pathlib import Path
from models.experimental import attempt_load
from classes import get_fog_type

os.environ['TZ'] = 'Asia/Jakarta'
time.tzset()


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

dataset_root = ROOT / 'dataset'
images = dataset_root / 'preprocessed_images'

WEIGHT = ROOT / 'weights'

lane_weight = WEIGHT / 'best-yolov5m-400epochs.pt'
vehicle_weight = WEIGHT / 'yolov5m-car-detection.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lane_model = attempt_load(lane_weight, map_location=device)
vehicle_model = attempt_load(vehicle_weight, map_location=device)

_type = 1

ip = '192.168.1.250'

s = socket.socket()
s.connect((ip, 8888))

for filename in os.listdir(images):
    img = cv2.imread(str(images / filename))
    Detection = get_fog_type(_type)
    # detection = Detection(img,device,lane_model,vehicle_model,s)
