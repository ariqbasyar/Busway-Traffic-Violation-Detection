import sys
import cv2
import socket
import time
import os
import torch

from pathlib import Path
from models.experimental import attempt_load
from classes import get_fog_type, INTEGER_TO_BYTES

os.environ['TZ'] = 'Asia/Jakarta'
time.tzset()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

dataset_root = ROOT / 'dataset'
images = dataset_root / 'preprocessed_images'

WEIGHT = ROOT / 'weights'

lane_weight = WEIGHT / 'yolov5m-lane-400epochs.pt'
vehicle_weight = WEIGHT / 'yolov5m-car-detection.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lane_model = attempt_load(lane_weight, map_location=device)
vehicle_model = attempt_load(vehicle_weight, map_location=device)

_type = 1 # see classes.get_fog_type

HOST = '127.0.0.1'
PORT = 5000

s = socket.socket()
s.connect((HOST, PORT))

filenames = os.listdir(images)

s.sendall((len(filenames)).to_bytes(*INTEGER_TO_BYTES))

for filename in filenames:
    print(filename)
    img = cv2.imread(str(images / filename))
    Detection = get_fog_type(_type)
    detection = Detection(img,device,lane_model,vehicle_model,s)
    detection.perform_detection()

s.close()
