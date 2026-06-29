import json
from fastapi import APIRouter, HTTPException
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference
from utils import decode_b64, binary_mask_to_frontend_rle

router = APIRouter()


@router.post("/predict-obj-refine", response_model=InferenceResponse)
async def predict_refine(req: InferenceRequest):
    image = decode_b64(req.image_base64)
    pre_defined_texts = req.pre_detect_labels if req.pre_detect_labels else ["person"]
    refine_texts = [
        p.text for p in req.prompts
        if p.text and p.text.lower() not in [t.lower() for t in pre_defined_texts]
    ]

    crop_cfg = req.crop_config.dict() if req.crop_config else None
    raw_results = inference.inference_with_obj_refine(
        image, refine_texts, pre_defined_texts,
        req.confidence_threshold, req.return_mask, req.merge_results, crop_cfg
    )

    if not raw_results:
        return InferenceResponse(results=[])

    final_list = []
    for r in raw_results:
        # 过滤掉 C++ 层附加的 __CROP__ 可视化标记框
        if r.class_name == "__CROP__":
            continue
        box_coords = [
            float(r.box.left), float(r.box.top),
            float(r.box.right), float(r.box.bottom)
        ]

        mask_rle = None
        mask_w = None
        mask_h = None
        if req.return_mask and r.segmentation is not None:
            mask = r.segmentation.mask
            mask_rle = binary_mask_to_frontend_rle(mask)
            if mask is not None:
                mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])

        final_list.append(DetectionResult(
            label=r.class_name,
            score=float(r.score),
            box=box_coords,
            mask=mask_rle if mask_rle else None,
            mask_width=mask_w,
            mask_height=mask_h,
        ))

    return InferenceResponse(results=final_list)
