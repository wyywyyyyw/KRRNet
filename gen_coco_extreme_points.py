import pycocotools.coco as cocoapi
import sys
import cv2
import numpy as np
import pickle
import json
SPLITS = ['train']
ANN_PATH = '../coco/annotations/instances_{}2017.json'
OUT_PATH = '../coco/annotations/instances_extreme_{}2017.json'
IMG_DIR = '../coco/{}2017/'
DEBUG = False
from scipy.spatial import ConvexHull

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

def _get_extreme_points(pts):
  l, t = min(pts[:, 0]), min(pts[:, 1])
  r, b = max(pts[:, 0]), max(pts[:, 1])
  # 3 degrees
  thresh = 0.02
  w = r - l + 1
  h = b - t + 1
  
  pts = np.concatenate([pts[-1:], pts, pts[:1]], axis=0)
  tl_distance = np.sqrt(np.square(pts[:, 0] - l) + np.square(pts[:, 1] - t))
  br_distance = np.sqrt(np.square(pts[:, 0] - r) + np.square(pts[:, 1] - b))
  tl_idx = np.argmin(tl_distance)
  br_idx = np.argmin(br_distance)
  tl_point = pts[tl_idx]
  br_point = pts[br_idx]

  tl_offset = [tl_point[0] - l, tl_point[1] - t]
  br_offset = [r - br_point[0], b - br_point[1]]

  return np.array([tl_point, br_point]), np.array(tl_offset), np.array(br_offset)

if __name__ == '__main__':
  for split in SPLITS:
    data = json.load(open(ANN_PATH.format(split), 'r'))
    coco = cocoapi.COCO(ANN_PATH.format(split))
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    num_classes = 80
    tot_box = 0
    print('num_images', num_images)
    anns_all = data['annotations']
    for i, ann in enumerate(anns_all):
      tot_box += 1
      bbox = ann['bbox']
      seg = ann['segmentation']
      if type(seg) == list:
        if len(seg) == 1:
          pts = np.array(seg[0]).reshape(-1, 2)
        else:
          pts = []
          for v in seg:
            pts += v
          pts = np.array(pts).reshape(-1, 2)
      else:
        mask = coco.annToMask(ann) * 255
        tmp = np.where(mask > 0)
        pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
      corner_point, tl_offset, br_offset = _get_extreme_points(pts)
      anns_all[i]['extreme_points'] = corner_point.astype(np.int32).copy().tolist()
      anns_all[i]['tl_offset'] = tl_offset.astype(np.int32).copy().tolist()
      anns_all[i]['br_offset'] = br_offset.astype(np.int32).copy().tolist()

      if DEBUG:
        img_id = ann['image_id']
        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_DIR.format(split) + img_info['file_name']
        img = cv2.imread(img_path)
        if type(seg) == list:
          mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
          cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], (255,0,0))
        else:
          mask = mask.reshape(img.shape[0], img.shape[1], 1)
        img = (0.4 * img + 0.6 * mask).astype(np.uint8)
        bbox = _coco_box_to_bbox(ann['bbox'])
        cl = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for j in range(corner_point.shape[0]):
          cv2.circle(img, (corner_point[j, 0], corner_point[j, 1]),
                          5, cl[j], -1)
        cv2.imshow('img', img)
        cv2.waitKey()
    print('tot_box', tot_box)   
    data['annotations'] = anns_all
    json.dump(data, open(OUT_PATH.format(split), 'w'))
  

