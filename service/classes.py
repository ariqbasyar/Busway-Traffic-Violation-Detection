import torch
import numpy as np

from torchvision.utils import save_image
from PIL import Image
from io import BytesIO
from time import time
from io import BytesIO
from collections import Counter
from model import (preprocess, detect, get_busway_box_from_prediction,
    get_middle, to_point, generate_center_from_rectangle)

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
        print('detect lane')
        pred = detect(self.lane_model,self.preprocessed)
        self.lane_box = get_busway_box_from_prediction(pred)
        print(self.lane_box.exterior.xy)

class OnlyDetectCar:
    def detect_car(self):
        print('detect car')
        # print(type(self.preprocessed))
        # print(self.preprocessed.shape)
        # filename = f'{np.random.randint(100)}.jpg'
        # print(filename)
        # save_image(self.preprocessed.cpu()[0], filename)
        pred = detect(self.vehicle_model,self.preprocessed)
        pred = [i for i in pred if i[6] != 'jalur_busway']
        print(pred)
        self.vehicle_labels = np.array(pred)[:,-1]
        self.vehicle_points = get_middle(pred)
        self.vehicle_points = to_point(self.vehicle_points)

class OnlyGetViolations:
    def get_violations(self):
        print('get violations')
        self.violating_vehicles = []
        for point,label in zip(self.vehicle_points,self.vehicle_labels):
            if label == 'bus': continue
            if self.lane_box.intersects(point):
                self.violating_vehicles.append(label)
        print(Counter(self.violating_vehicles))
        self.end = time()

class OnlySendPreprocess:
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
    def __init__(self, raw_data, **kwargs):
        try:
            data = BytesIO(raw_data)
            data.seek(0)
            data = np.load(data,allow_pickle=True)
            data = torch.from_numpy(data).to(kwargs.get('device'))
        except ValueError:
            data = None
        super().__init__(None,preprocessed=data,**kwargs)

    def perform_detection(self):
        self.detect_car()
        self.detect_lane()
        self.get_violations()

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
