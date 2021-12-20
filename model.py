import cv2
import numpy as np
import torch

from utils.plots import Annotator, colors
from utils.general import non_max_suppression
from shapely.geometry import MultiPoint, Point

def preprocess(main_img, device):
  img = cv2.resize(main_img, (512,512))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = np.moveaxis(img,-1,0)
  img = torch.from_numpy(img).to(device)
  img = img.float()/255.0  # 0 - 255 to 0.0 - 1.0
  if img.ndimension() == 3:
      img = img.unsqueeze(0)
  return img

def detect(model,img):
  """
    Perform prediction of an image with given torch model

    :param torch.nn.Module model: prediction model
    :param PIL.Image.Image img: an image to predict
    :return: an array of predictions [x0,y0,x1,y1,conf,index_label]
  """
  pred = model(img, augment=False)[0]
  pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.25)
  labels = model.names

  items = []
  if len(pred) and pred[0] is not None:
    for p in pred[0]:
      row = []
      x0,y0,x1,y1 = p[:4].tolist()
      conf = float(p[4])
      idx_label = int(p[-1])
      row = [x0,y0,x1,y1,conf,idx_label,labels[idx_label]]
      items.append(row)
  return items

def box_label(pred,img,show_label=True):
  """
    Make a box label for an image with
    given prediction [x0,y0,x1,y1,conf,index_label]
    and labels [label1,label2,...]

    :param list pred: an array of n predictions * prediction
      [x0,y0,x1,y1,conf,index_label]
    :param Image.Image/array img: the image for the prediction
    :param [str] labels: labels of prediction [label1,label2,...]
    :return: the labeled image
  """
  for p in pred:
    annotator = Annotator(img)
    box = tuple(p[:4])
    conf = p[4]
    c = int(p[5])
    label = p[-1]
    text = f'{label} {conf:.2f}'
    if show_label == False:
        text = ''
    annotator.box_label(box, text, colors(c, True))
  return annotator.result()

def get_middle(pred):
  X = []
  for p in pred:
    x0,y0,x1,y1 = p[:4]
    x = x0 + (x1 - x0)/2
    y = y0 + (y1 - y0)/2
    X.append([x,y])
  return np.array(X,dtype=int)

def get_busway_box_from_prediction(pred):
  n_labels_predicted = len(np.unique(np.array(pred)[:,-1]))
  if n_labels_predicted == 2:
    X = get_middle(pred)

    convex_hull = MultiPoint(X).convex_hull
    return convex_hull
  return None

def to_point(points):
    res = []
    for point in points:
        res.append(Point(*point))
    return res

def generate_center_from_rectangle(x0, y0, x1, y1):
    return (x0+(x1-x0)/2, y0+(y1-y0)/2)
