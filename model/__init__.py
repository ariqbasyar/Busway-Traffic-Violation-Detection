from shapely.geometry import Point
from model.models.experimental import attempt_load
from model.model import (preprocess, get_middle,
    detect, box_label, get_busway_box_from_prediction)

def to_point(points):
    res = []
    for point in points:
        res.append(Point(*point))
    return res
