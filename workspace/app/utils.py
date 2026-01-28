import cv2
import numpy as np
import os
import base64
from typing import Dict


import numpy as np

def calculate_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def get_union_box(box1, box2):
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    ]

def to_square(box, img_w, img_h):
    """将矩形框扩展为正方形，并确保不越界"""
    l, t, r, b = box
    w = r - l
    h = b - t
    side = max(w, h)
    
    center_x, center_y = l + w / 2, t + h / 2
    
    new_l = max(0, center_x - side / 2)
    new_t = max(0, center_y - side / 2)
    new_r = min(img_w, new_l + side)
    new_b = min(img_h, new_t + side)
    
    # 再次修正因越界导致的非正方形
    return [new_l, new_t, new_r, new_b]

def get_iou(box1, box2):
    """计算两个框的交并比"""
    inter_l = max(box1[0], box2[0])
    inter_t = max(box1[1], box2[1])
    inter_r = min(box1[2], box2[2])
    inter_b = min(box1[3], box2[3])
    
    inter_area = max(0, inter_r - inter_l) * max(0, inter_b - inter_t)
    if inter_area <= 0: return 0
    
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    return inter_area / float(area1 + area2 - inter_area)

def merge_person_boxes(person_boxes, img_w, img_h, max_area_ratio=0.4, min_area_ratio=0.01, dist_threshold=100):
    if not person_boxes: return []

    max_area = (img_w * img_h) * max_area_ratio
    min_area = (img_w * img_h) * min_area_ratio

    # 1. 初始框提取（带 Padding）
    active_boxes = []
    for p in person_boxes:
        # 统一使用底层属性访问
        b = [float(p.box.left), float(p.box.top), float(p.box.right), float(p.box.bottom)]
        w, h = b[2] - b[0], b[3] - b[1]
        if calculate_area(b) < min_area:
            continue
        pad_w, pad_h = w * 0.15, h * 0.15
        active_boxes.append([
            max(0, b[0]-pad_w), max(0, b[1]-pad_h), 
            min(img_w, b[2]+pad_w), min(img_h, b[3]+pad_h)
        ])

    # 2. 贪心合并
    merged = []
    used = set()
    for i in range(len(active_boxes)):
        if i in used: continue
        curr = active_boxes[i]
        used.add(i)
        
        changed = True
        while changed:
            changed = False
            for j in range(len(active_boxes)):
                if j in used: continue
                
                # 距离判断
                dx = max(0, curr[0] - active_boxes[j][2], active_boxes[j][0] - curr[2])
                dy = max(0, curr[1] - active_boxes[j][3], active_boxes[j][1] - curr[3])
                
                if max(dx, dy) < dist_threshold:
                    union = get_union_box(curr, active_boxes[j])
                    # 面积限制 + 长宽比限制（防止合并成极细长条）
                    u_w, u_h = union[2]-union[0], union[3]-union[1]
                    aspect_ratio = max(u_w, u_h) / (min(u_w, u_h) + 1e-6)
                    
                    if calculate_area(union) <= max_area and aspect_ratio < 3.0:
                        curr = union
                        used.add(j)
                        changed = True
        merged.append(curr)

    # 3. 正方形化处理
    squared_regions = [to_square(b, img_w, img_h) for b in merged]

    # 4. 最终去重：如果两个正方形 Crop 高度重叠 (IOU > 0.5)，则合并它们
    final_regions = []
    already_processed = set()
    for i in range(len(squared_regions)):
        if i in already_processed: continue
        curr = squared_regions[i]
        for j in range(i + 1, len(squared_regions)):
            if j in already_processed: continue
            if get_iou(curr, squared_regions[j]) > 0.5:
                curr = get_union_box(curr, squared_regions[j])
                already_processed.add(j)
        final_regions.append(curr)
        
    return final_regions


def decode_b64(b64_str):
    try:
        data = base64.b64decode(b64_str.split(',')[-1])
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail="Base64 decode failed")

def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    使用 Numpy 向量化加速 RLE 编码。
    """
    if mask is None: return None
    
    # 扁平化
    flat = mask.ravel(order='F')
    flat = (flat > 0).astype(np.int8)
    
    if len(flat) == 0:
        return {'size': list(mask.shape), 'counts': []}

    diffs = np.where(flat[1:] != flat[:-1])[0] + 1
    bounds = np.concatenate(([0], diffs, [len(flat)]))
    counts = np.diff(bounds)
    counts_list = counts.tolist()
    
    if flat[0] == 1:
        counts_list = [0] + counts_list
        
    return {'size': list(mask.shape), 'counts': counts_list}