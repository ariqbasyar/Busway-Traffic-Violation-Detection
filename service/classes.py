import numpy as np

from time import time

import torch

from model import (preprocess, detect, get_busway_box_from_prediction,
    get_middle, to_point)

BYTES_ARRAY = 16
ENDIAN = 'little'
INTEGER_TO_BYTES = (BYTES_ARRAY,ENDIAN)

class BaseDetection:
    def __init__(self,img,device=None,lane_model=None,vehicle_model=None,
            socket=None,start=None):
        self.img = img
        self.device = device
        if not device:
            self.device = torch.device('cpu')
        self.lane_model = lane_model
        self.vehicle_model = vehicle_model
        self.socket = socket
        if start is None:
            self.start = int(time()*1E3) # integer in milliseconds
        else:
            self.start = start

        self.preprocessed = None
        self.lane_box = None
        self.vehicle_labels = None
        self.vehicle_points = None
        self.count_violations = -1
        self.end = None

    def perform_detection(self):
        self.preprocess()
        self.detect_lane()
        self.detect_car()
        self.get_violations()
        self.send()

class OnlyPreprocess:
    def preprocess(self):
        self.preprocessed = preprocess(self.img,self.device)

class OnlyDetectLane:
    def detect_lane(self):
        pred = detect(self.lane_model,self.preprocessed)
        self.lane_box = get_busway_box_from_prediction(pred)

class OnlyDetectCar:
    def detect_car(self):
        pred = detect(self.vehicle_model,self.preprocessed)
        self.vehicle_labels = np.array(pred)[:,-1]
        self.vehicle_points = get_middle(pred)
        self.vehicle_points = to_point(self.vehicle_points)

class OnlyGetViolations:
    def get_violations(self):
        self.count_violations = 0
        for point,label in zip(self.vehicle_points,self.vehicle_labels):
            if label == 'bus': continue
            if self.lane_box.intersects(point):
                self.count_violations += 1
        self.end = time()

class OnlySendPreprocess:
    def send(self):
        payload_time = self.start.to_bytes(*INTEGER_TO_BYTES)

        data = self.preprocessed
        if self.device.type == 'cuda':
            data = self.preprocessed.cpu()
        arr_preprocessed = np.array(data).tobytes()
        size_preprocessed = len(arr_preprocessed)
        print(size_preprocessed)
        payload_image = size_preprocessed.to_bytes(*INTEGER_TO_BYTES)\
            + arr_preprocessed

        payload = payload_time + payload_image
        self.socket.sendall(payload)

class FogOnlyPreprocessing(BaseDetection,
                            OnlyPreprocess,
                            OnlySendPreprocess):
    def perform_detection(self):
        self.preprocess()
        self.send()

class ServerOnlyDetect(BaseDetection,
                        OnlyDetectLane,
                        OnlyDetectCar,
                        OnlyGetViolations):
    pass

_types = {
    1: (FogOnlyPreprocessing,ServerOnlyDetect),
}

def get_fog_type(_type):
    classes = _types.get(_type)
    if classes is None: return None
    return classes[0]

def get_server_type(_type):
    classes = _types.get(_type)
    if classes is None: return None
    return classes[1]
