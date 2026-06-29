import cv2, numpy as np, json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models import InferenceRequest, InferenceResponse, DetectionResult
import inference
from utils import decode_b64, binary_mask_to_frontend_rle

router = APIRouter()


def _raw_to_detection_results(raw_results, return_mask: bool) -> List[DetectionResult]:
    """将 C++ 返回的原始结果统一转换为前端可用的 DetectionResult"""
    final_list = []
    if not raw_results:
        return final_list

    for r in raw_results:
        # 过滤掉 C++ 层附加的 __CROP__ 可视化标记框
        if getattr(r, "class_name", "") == "__CROP__":
            continue

        box_coords = [
            float(r.box.left),
            float(r.box.top),
            float(r.box.right),
            float(r.box.bottom),
        ]

        mask_rle = None
        mask_w = None
        mask_h = None
        if return_mask and r.segmentation is not None:
            mask = r.segmentation.mask
            mask_rle = binary_mask_to_frontend_rle(mask)
            if mask is not None:
                # OpenCV Mat shape 为 (height, width)
                mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])

        final_list.append(
            DetectionResult(
                label=r.class_name,
                score=float(r.score),
                box=box_coords,
                mask=mask_rle if mask_rle else None,
                mask_width=mask_w,
                mask_height=mask_h,
            )
        )

    return final_list


def _decode_uploaded_image(file: UploadFile) -> np.ndarray:
    """将上传文件解码为 OpenCV BGR 图像"""
    try:
        contents = file.file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


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
        if res:
            final_raw.extend(res)

    # 2. 遍历处理“混合提示”或“纯 Box 提示”（必须一一对应）
    for unit in req.prompts:
        if not unit.boxes:
            continue
        boxes_dict = [
            {"type": b.label, "x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]}
            for b in unit.boxes
        ]
        if unit.text:
            res = inference.inference_with_mixed_prompt(
                image, unit.text, boxes_dict, req.confidence_threshold, req.return_mask
            )
        else:
            res = inference.inference_with_box_prompt(
                image, boxes_dict, req.confidence_threshold, req.return_mask
            )

        if res:
            final_raw.extend(res)

    return InferenceResponse(results=_raw_to_detection_results(final_raw, req.return_mask))


@router.post("/predict/file", response_model=InferenceResponse)
async def predict_file(
    mode: str = Form("multi-class"),
    image: UploadFile = File(...),
    class_names: List[str] = Form(default=[]),
    target_boxes: Optional[str] = Form(None),
    prompt_image: Optional[UploadFile] = File(None),
    prompt_boxes: Optional[str] = Form(None),
    pre_detect_labels: Optional[str] = Form("person"),
    merge_results: bool = Form(True),
    crop_config_json: Optional[str] = Form(None),
    confidence: float = Form(0.3),
    return_mask: bool = Form(True),
):
    """
    文件上传版统一推理接口，前端直接上传图片并获取 JSON 结果（含 RLE mask）。

    mode 可选值：
      - multi-class : 多类别文本提示（依赖 class_names）
      - box         : 纯几何框提示（依赖 target_boxes）
      - mixed       : 文本 + 几何框提示（依赖 class_names + target_boxes）
      - from-image  : 跨图几何提示（依赖 prompt_image + prompt_boxes）
      - obj-refine  : 先 crop 再精细识别（依赖 class_names / pre_detect_labels / crop_config_json）
    """
    target_image = _decode_uploaded_image(image)
    raw_results = []

    try:
        if mode == "multi-class":
            prompts = [c.strip() for c in class_names if c.strip()]
            if not prompts:
                raise HTTPException(status_code=400, detail="class_names is required for multi-class mode")
            raw_results = inference.inference_with_multi_class_prompt(
                target_image, prompts, confidence, return_mask
            ) or []

        elif mode == "box":
            boxes = json.loads(target_boxes) if target_boxes else []
            if not boxes:
                raise HTTPException(status_code=400, detail="target_boxes is required for box mode")
            raw_results = inference.inference_with_box_prompt(
                target_image, boxes, confidence, return_mask
            ) or []

        elif mode == "mixed":
            text = class_names[0].strip() if class_names else ""
            boxes = json.loads(target_boxes) if target_boxes else []
            if not text and not boxes:
                raise HTTPException(status_code=400, detail="class_names or target_boxes is required for mixed mode")
            raw_results = inference.inference_with_mixed_prompt(
                target_image, text, boxes, confidence, return_mask
            ) or []

        elif mode == "from-image":
            if prompt_image is None:
                raise HTTPException(status_code=400, detail="prompt_image is required for from-image mode")
            prompt_img = _decode_uploaded_image(prompt_image)
            boxes = json.loads(prompt_boxes) if prompt_boxes else []
            if not boxes:
                raise HTTPException(status_code=400, detail="prompt_boxes is required for from-image mode")
            geom_label = inference.setup_image_prompt(prompt_img, boxes)
            if not geom_label:
                raise HTTPException(status_code=500, detail="Failed to setup geometry prompt")
            raw_results = inference.inference_with_image_prompt(
                target_image, geom_label, confidence, return_mask
            ) or []

        elif mode == "obj-refine":
            refine_texts = [c.strip() for c in class_names if c.strip()]
            pre_labels = [p.strip() for p in pre_detect_labels.split(",") if p.strip()] if pre_detect_labels else ["person"]
            if not refine_texts:
                raise HTTPException(status_code=400, detail="class_names is required for obj-refine mode")

            crop_config = None
            if crop_config_json:
                try:
                    crop_config = json.loads(crop_config_json)
                except Exception:
                    pass

            raw_results = inference.inference_with_obj_refine(
                target_image,
                refine_texts,
                pre_labels,
                confidence,
                return_mask,
                merge_results,
                crop_config,
            ) or []

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return InferenceResponse(results=_raw_to_detection_results(raw_results, return_mask))
