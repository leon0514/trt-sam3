import cv2
import numpy as np
import os
import base64
from typing import Dict
from fastapi import APIRouter, HTTPException

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

def merge_person_boxes(person_boxes, img_w, img_h, max_area_ratio=0.9999999, min_area_ratio=0.0000003, dist_threshold=100):
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


def get_person_regions(person_boxes, img_w, img_h, min_area_ratio=0.0000003):
    """
    直接返回所有符合面积要求的 person 框，并应用 15% 的 padding
    """
    if not person_boxes:
        return []

    min_area = (img_w * img_h) * min_area_ratio
    final_regions = []

    for p in person_boxes:
        # 统一访问框的坐标（根据你的描述，假设 p.box 具有 left, top, right, bottom 属性）
        # 如果 p 只是字典，请改为: b = [p['box'][0], p['box'][1], p['box'][2], p['box'][3]]
        b = [float(p.box.left), float(p.box.top), float(p.box.right), float(p.box.bottom)]

        # 计算原始面积
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < min_area:
            continue

        # 应用 15% 的 padding
        w, h = b[2] - b[0], b[3] - b[1]
        pad_w, pad_h = w * 0.15, h * 0.15

        # 裁剪边界到图像内
        region = [
            max(0, b[0] - pad_w),
            max(0, b[1] - pad_h),
            min(img_w, b[2] + pad_w),
            min(img_h, b[3] + pad_h)
        ]

        final_regions.append(region)

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


def box_iou(box1, box2):
    """
    计算两个 Box 的交并比 (Intersection over Union)
    box1, box2: C++ 绑定的 Box 对象，包含 left, top, right, bottom 属性
    """
    # 1. 计算交集区域的坐标
    inter_left = max(box1.left, box2.left)
    inter_top = max(box1.top, box2.top)
    inter_right = min(box1.right, box2.right)
    inter_bottom = min(box1.bottom, box2.bottom)
    
    # 2. 计算交集区域的宽和高 (如果两个框不相交，宽或高可能为负数，用 max(0, x) 处理)
    inter_w = max(0.0, inter_right - inter_left)
    inter_h = max(0.0, inter_bottom - inter_top)
    
    # 3. 计算交集面积
    inter_area = inter_w * inter_h
    
    # 4. 计算两个框各自的面积
    area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
    area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
    
    # 5. 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 防止除以 0
    if union_area <= 0:
        return 0.0
        
    # 6. 返回 IOU
    return inter_area / union_area

def nms(boxes, iou_threshold=0.5):
    """
    支持多类别的非极大值抑制
    """
    if not boxes:
        return []
    
    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
    keep = []
    
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        # 过滤条件：
        # 如果类别不同，直接保留 (返回 True)
        # 如果类别相同，则判断 IOU 是否小于阈值，小于阈值才保留
        boxes = [
            b for b in boxes 
            if (current.class_name != b.class_name) or (box_iou(current.box, b.box) < iou_threshold)
        ]
        
    return keep