import cv2, numpy as np
from fastapi import APIRouter, HTTPException
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference
from utils import decode_b64
from utils import binary_mask_to_rle

router = APIRouter()

@router.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    image = decode_b64(req.image_base64)
    final_raw = []
    
    # 1. 提取所有“仅有文本”的 Prompt，进行聚合推理
    pure_texts = [p.text for p in req.prompts if p.text and not p.boxes]
    if pure_texts:
        res = inference.inference_with_multi_class_prompt(
            image, pure_texts, req.confidence_threshold, req.return_mask
        )
        if res: final_raw.extend(res)

    # 2. 遍历处理“混合提示”或“纯 Box 提示”（必须一一对应）
    for unit in req.prompts:
        if not unit.boxes:
            continue
        boxes_dict = [
            {"type": b.label, "x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]} 
            for b in unit.boxes
        ]
        # 混合提示 or 纯 Box 提示
        if unit.text:
            res = inference.inference_with_mixed_prompt(
                image, unit.text, boxes_dict, req.confidence_threshold, req.return_mask
            )
        else:
            res = inference.inference_with_box_prompt(
                image, boxes_dict, req.confidence_threshold, req.return_mask
            )
        
        if res: final_raw.extend(res)

    # 3. 格式转换（过滤掉 C++ 层附加的 __CROP__ 可视化标记框）
    final_list = []
    for r in final_raw:
        if r.class_name == "__CROP__":
            continue
        box_coords = [float(r.box.left), float(r.box.top), float(r.box.right), float(r.box.bottom)]
        final_list.append(DetectionResult(
            label=r.class_name,
            score=float(r.score),
            box=box_coords,
            mask=binary_mask_to_rle(r.segmentation.mask) if (req.return_mask and r.segmentation) else None
        ))
    
    return InferenceResponse(results=final_list)