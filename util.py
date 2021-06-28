import json
import pathlib
import time
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(train_data_path):
    img_dir = train_data_path[0]
    gt_dir = train_data_path[1]
    img_fps = glob.glob(os.path.join(img_dir, "*"))
    gt_fps = []

    train_data = []

    for img_fp in img_fps:
        img_id = img_fp.split("/")[-1].split(".")[0]
        gt_fn = "gt_{}.txt".format(img_id)
        gt_fp = os.path.join(gt_dir, gt_fn)

        assert os.path.exists(img_fp)
        gt_fps.append(gt_fp)
        
        img_path = pathlib.Path(img_fp)
        label_path = pathlib.Path(gt_fp)
        
        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
            train_data.append((str(img_path), str(label_path)))
    
    assert len(img_fps) == len(gt_fps)

    return train_data


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def save_result(result_path, box_list, score_list, is_output_polygon):
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)
