import cv2, numpy as np
from fastapi import APIRouter, HTTPException
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference

from utils import decode_b64

router = APIRouter()

@router.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    image = decode_b64(req.image_base64)
    final_raw = []
    
    for unit in req.prompts:
        if unit.text and not unit.boxes:
            res = inference.inference_with_multi_class_prompt(image, [unit.text], req.confidence_threshold, req.return_mask)
        elif unit.boxes and not unit.text:
            boxes_dict = [{"type": b.label, "x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]} for b in unit.boxes]
            res = inference.inference_with_box_prompt(image, boxes_dict, req.confidence_threshold, req.return_mask)
        else:
            boxes_dict = [{"type": b.label, "x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]} for b in unit.boxes]
            res = inference.inference_with_mixed_prompt(image, unit.text, boxes_dict, req.confidence_threshold, req.return_mask)
        
        if res: final_raw.extend(res)

    results = [DetectionResult(label=r.label, score=r.score, box=r.box) for r in final_raw]
    return InferenceResponse(results=results)