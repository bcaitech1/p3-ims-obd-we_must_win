import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO 
import numpy as np
import matplotlib.pyplot as plt 
import cv2

COLORS = [
    (204,24,30),
    (39,147,232),
    (85,153,0),
    (102,102,102),
    (27,133,184),
    (90,82,85),
    (85,158,131),
    (174,90,65),
    (195,203,113),
    (243,119,54),
    (184,167,234)
]

# Felzenszwalb et al
def non_max_suppression_fast(boxes, scores, iou_threshold):
    
    '''
        boxes : coordinates of each box
        scores : score of each box
        iou_threshold : iou threshold(box with iou larger than threshold will be removed)
    '''
    
    if len(boxes) == 0:
        return []

    # Init the picked box info
    pick = []

    # Box coordinate consist of left top and right bottom
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute area of each boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Greedily select the order of box to compare iou
    idxs = np.argsort(scores)

    while(len(idxs) > 0):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # With vector implementation, we can calculate fast
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h

        # Calculate the iou
        iou = intersection / (area[idxs[:last]] + area[idxs[last]] - intersection)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))

    return boxes[pick].astype("int")


def draw_result(output, idx):
    out = output[idx]
    coco = COCO(annotation_file='../input/data/val.json')
    data_dir = '../input/data/'
    idx = coco.getImgIds(idx)
    img = cv2.imread(data_dir+coco.load_imgs(idx)[0]['file_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_img = img.copy()

    anns = coco.load_anns(coco.get_ann_ids(idx))
    for ann in anns:
        cls_id = ann['category_id']
        x,y,w,h = [int(i) for i in ann['bbox']]
        img = cv2.rectangle(img, (x,y), (x+w, y+h), COLORS[cls_id], 3)
        text = coco.cats[cls_id]['name']
        cv2.putText(img, text, (x+10,y+19), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0))

    for c,bboxes in enumerate(out):
#         bboxes = non_max_suppression_fast(bboxes[:,:4], bboxes[:,-1], 0.00000000007)
        for bbox in bboxes:
#             print(bbox)
            x1,y1,x2,y2 = bbox[:-1].astype('int64')
            score = str(bbox[-1])[:5]
#             score = ''
            pred_img = cv2.rectangle(pred_img, (x1,y1), (x2,y2), COLORS[c], 3 )
            text = coco.cats[c]['name'] + ":" + score
            cv2.putText(pred_img, text, (x1,y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0))
            print(text)


    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(20,20))
    ax[0].imshow(img)
    ax[1].imshow(pred_img)
    plt.show()