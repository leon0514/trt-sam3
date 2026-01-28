import os, json, uuid, shutil, cv2
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference
from utils import decode_b64
from utils import binary_mask_to_rle

router = APIRouter()

@router.post("/predict-person-about-small-object", response_model=InferenceResponse)
async def predict_person_refine(req: InferenceRequest):
    image = decode_b64(req.image_base64)
    refine_texts = [p.text for p in req.prompts if p.text and p.text.lower() != 'person']
    raw_results = inference.inference_with_person_refine(image, refine_texts, req.confidence_threshold, req.return_mask)
    
    if not raw_results:
        return InferenceResponse(results=[])

    # 2. 关键步骤：手动转换为 Pydantic 定义的 DetectionResult 列表
    final_list = []
    for r in raw_results:
        # 根据报错信息 r.box 是一个可以通过索引访问的对象，或者具有 left, top 等属性
        # 这里使用通用的处理方式，确保转换为 list[float]
        try:
            # 如果 r.box 本身就是类似列表的对象 [x1, y1, x2, y2]
            box_coords = [float(r.box[0]), float(r.box[1]), float(r.box[2]), float(r.box[3])]
        except:
            # 如果 r.box 是具有 .left, .top 属性的对象
            box_coords = [float(r.box.left), float(r.box.top), float(r.box.right), float(r.box.bottom)]

        final_list.append(DetectionResult(
            label=r.class_name,
            score=float(r.score),
            box=box_coords,
            mask=binary_mask_to_rle(r.segmentation.mask) if (req.return_mask and r.segmentation) else None
        ))
    
    return InferenceResponse(results=final_list)