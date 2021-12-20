import torch
import psutil
import GPUtil
import numpy as np

from torchvision.utils import save_image
from PIL import Image
from io import BytesIO
from time import time
from io import BytesIO
from collections import Counter
from model import (preprocess, detect, get_busway_box_from_prediction,
    get_middle, to_point)

BYTES_ARRAY = 16
ENDIAN = 'little'
INTEGER_TO_BYTES = (BYTES_ARRAY,ENDIAN)

class BaseDetection:
    def __init__(self,img,device=None,lane_model=None,vehicle_model=None,
            socket=None,start=None,preprocessed=None):
        self.img = img
        self.preprocessed = preprocessed
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

        self.lane_box = None
        self.vehicle_labels = None
        self.vehicle_points = None
        self.count_violations = -1
        self.end = None
        self.cpu_util = None
        self.mem_util = None

    def get_data(self):
        print(f'{self.start},{self.end}')

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
        pred = [i for i in pred if i[6] != 'jalur_busway' and i[6] != 'bus']
        if len(pred) == 0:
            self.vehicle_labels = []
            self.vehicle_points = []
            return
        self.vehicle_labels = np.array(pred)[:,-1]
        self.vehicle_points = get_middle(pred)
        self.vehicle_points = to_point(self.vehicle_points)

class OnlyGetViolations:
    def get_violations(self):
        self.violating_vehicles = []
        if self.lane_box is None:
            return
        for point,label in zip(self.vehicle_points,self.vehicle_labels):
            if label == 'bus': continue
            if self.lane_box.intersects(point):
                self.violating_vehicles.append(label)

class FogOnlyPreprocessing(BaseDetection,
                            OnlyPreprocess):
    def send(self):
        payload_time = self.start.to_bytes(*INTEGER_TO_BYTES)

        data = self.preprocessed
        if self.device.type == 'cuda':
            data = self.preprocessed.cpu()
        arr_preprocessed = np.array(data)
        f = BytesIO()
        np.save(f,arr_preprocessed)
        f.seek(0)
        out = f.read()
        size_preprocessed = len(out)
        payload_image = size_preprocessed.to_bytes(*INTEGER_TO_BYTES) + out

        payload = payload_time + payload_image
        self.socket.sendall(payload)

    def perform_detection(self):
        self.preprocess()
        self.send()

class ServerOnlyDetect(BaseDetection,
                        OnlyDetectLane,
                        OnlyDetectCar,
                        OnlyGetViolations):
    def __init__(self, raw_data, **kwargs):
        try:
            data = BytesIO(raw_data)
            data.seek(0)
            data = np.load(data,allow_pickle=True)
            data = torch.from_numpy(data).to(kwargs.get('device'))
        except:
            data = None
        super().__init__(None,preprocessed=data,**kwargs)

    def get_violations(self):
        super().get_violations()
        self.end = int(time()*1E3)

    def perform_detection(self):
        self.detect_car()
        self.detect_lane()
        self.get_violations()

class FogFullDetection(BaseDetection,
                        OnlyPreprocess,
                        OnlyDetectCar,
                        OnlyDetectLane,
                        OnlyGetViolations):
    def send(self):
        payload_time = self.start.to_bytes(*INTEGER_TO_BYTES)

        payload_violation = np.array(self.violating_vehicles)
        f = BytesIO()
        np.save(f,payload_violation)
        f.seek(0)
        out = f.read()
        size_payload_violation = len(out)
        payload_violation_final =\
            size_payload_violation.to_bytes(*INTEGER_TO_BYTES) + out

        payload = payload_time + payload_violation_final
        self.socket.sendall(payload)

    def perform_detection(self):
        self.preprocess()
        self.detect_car()
        self.detect_lane()
        self.get_violations()
        self.send()

class ServerRecvViolation(BaseDetection):

    def __init__(self, raw_data, **kwargs):
        try:
            data = list(raw_data)
        except:
            data = None
        super().__init__(data, **kwargs)
        # print(data)

        self.end = int(time()*1E3)

    def perform_detection(self):
        pass

class FogOnlySend(BaseDetection):

    def send(self):
        payload_time = self.start.to_bytes(*INTEGER_TO_BYTES)

        f = BytesIO()
        np.save(f,self.img)
        f.seek(0)
        out = f.read()
        size_img = len(out)
        payload_image = size_img.to_bytes(*INTEGER_TO_BYTES) + out

        payload = payload_time + payload_image
        self.socket.sendall(payload)

    def perform_detection(self):
        self.send()

class ServerFullDetection(BaseDetection,
                            OnlyPreprocess,
                            OnlyDetectCar,
                            OnlyDetectLane,
                            OnlyGetViolations):

    def __init__(self, raw_data, **kwargs):
        try:
            data = BytesIO(raw_data)
            data.seek(0)
            data = np.load(data,allow_pickle=True)
        except:
            data = None
        super().__init__(data,**kwargs)

    def get_violations(self):
        super().get_violations()
        self.end = int(time()*1E3)

    def perform_detection(self):
        self.preprocess()
        self.detect_car()
        self.detect_lane()
        self.get_violations()

_types = {
    1: (FogOnlyPreprocessing,ServerOnlyDetect),
    2: (FogFullDetection,ServerRecvViolation),
    3: (FogOnlySend,ServerFullDetection),
}

def get_fog_type(_type):
    classes = _types.get(_type)
    if classes is None: return None
    return classes[0]

def get_server_type(_type):
    classes = _types.get(_type)
    if classes is None: return None
    return classes[1]
