import os, json, uuid, shutil, cv2
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference
from utils import decode_b64
from utils import binary_mask_to_rle

router = APIRouter()

@router.post("/predict-obj-refine", response_model=InferenceResponse)
async def predict_refine(req: InferenceRequest):
    image = decode_b64(req.image_base64)
    pre_defined_texts = req.pre_detect_labels if req.pre_detect_labels else ["person"]
    refine_texts = [p.text for p in req.prompts if p.text and p.text.lower() not in [t.lower() for t in pre_defined_texts]]

    crop_cfg = req.crop_config.dict() if req.crop_config else None
    raw_results = inference.inference_with_obj_refine(
        image, refine_texts, pre_defined_texts,
        req.confidence_threshold, req.return_mask, req.merge_results, crop_cfg
    )
    
    if not raw_results:
        return InferenceResponse(results=[])

    # 2. 关键步骤：手动转换为 Pydantic 定义的 DetectionResult 列表
    # 过滤掉 C++ 层附加的 __CROP__ 可视化标记框
    final_list = []
    for r in raw_results:
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