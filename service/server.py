import socket
import time
import os
import numpy as np
import torch

from pathlib import Path
from models.experimental import attempt_load
from classes import BYTES_ARRAY, ENDIAN, get_server_type

os.environ['TZ'] = 'Asia/Jakarta'
time.tzset()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

dataset_root = ROOT / 'dataset'
images = dataset_root / 'preprocessed_images'

WEIGHT = ROOT / 'weights'

lane_weight = WEIGHT / 'best-yolov5m-400epochs.pt'
vehicle_weight = WEIGHT / 'yolov5m-car-detection.pt'

print('Loading model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lane_model = attempt_load(lane_weight, map_location=device)
vehicle_model = attempt_load(vehicle_weight, map_location=device)

HOST = '192.168.1.250'
PORT = 5000

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

_type = 1 # see classes.get_server_type

conn,addr = s.accept()
print(f'Connection from {addr}')

data = b''
while len(data) < BYTES_ARRAY:
    _data = conn.recv(4096)
    data += _data
amount_of_files = data[:BYTES_ARRAY]
amount_of_files = int.from_bytes(amount_of_files, ENDIAN)
print('amount of files')
print(amount_of_files)

data = data[BYTES_ARRAY:]

for i in range(amount_of_files):
    print('='*50)
    # get the time
    while len(data) < BYTES_ARRAY:
        _data = conn.recv(4096)
        data += _data
    time_payload = data[:BYTES_ARRAY]
    time_payload = int.from_bytes(time_payload,ENDIAN)
    print(time_payload)

    data = data[BYTES_ARRAY:]

    # get the size of the next payload
    while len(data) < BYTES_ARRAY:
        _data = conn.recv(4096)
        data += _data
    next_size = data[:BYTES_ARRAY]
    next_size = int.from_bytes(next_size,ENDIAN)

    data = data[BYTES_ARRAY:]

    next_data = b''

    # get the next payload
    if (next_size > 0):
        while len(data) < next_size:
            _data = conn.recv(4096)
            data += _data
        next_data = data[:next_size]

        data = data[next_size:]

    Detection = get_server_type(_type)
    detection = Detection(next_data, start=time_payload, device=device,
                        lane_model=lane_model,vehicle_model=vehicle_model)
    detection.perform_detection()

    del Detection, detection, next_data
