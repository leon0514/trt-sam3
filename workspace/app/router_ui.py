import os, json, uuid, shutil, cv2
from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
import inference

router = APIRouter()

@router.post("/process-image")
async def process_image(
    mode: str = Form(...),
    target_image: UploadFile = File(...),
    text_prompts: Optional[str] = Form(None),
    prompt_image: Optional[UploadFile] = File(None),
    target_boxes: Optional[str] = Form(None),
    prompt_boxes: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.5),
    return_mask: bool = Form(True)
):
    # 临时保存上传文件
    target_uuid = uuid.uuid4()
    target_path = os.path.join(inference.UPLOADS_DIR, f"{target_uuid}_target.jpg")
    with open(target_path, "wb") as f: 
        shutil.copyfileobj(target_image.file, f)
    
    prompt_path = None
    if prompt_image and prompt_image.filename:
        prompt_path = os.path.join(inference.UPLOADS_DIR, f"{uuid.uuid4()}_prompt.jpg")
        with open(prompt_path, "wb") as f: 
            shutil.copyfileobj(prompt_image.file, f)

    try:
        t_boxes = json.loads(target_boxes) if target_boxes else []
        p_boxes = json.loads(prompt_boxes) if prompt_boxes else []
        output_path = None

        if mode == "multi-class":
            prompts = [p.strip() for p in text_prompts.split(',') if p.strip()] if text_prompts else []
            output_path = inference.run_multi_class_prompt(target_path, prompts, confidence_threshold, return_mask)
            
        elif mode == "box":
            output_path = inference.run_box_prompt(target_path, t_boxes, confidence_threshold, return_mask)
            
        elif mode == "mixed":
            text = text_prompts.split(',')[0].strip() if text_prompts else ""
            output_path = inference.run_mixed_prompt(target_path, text, t_boxes, confidence_threshold, return_mask)
            
        elif mode == "from-image":
            output_path = inference.run_from_image_prompt(target_path, prompt_path, p_boxes, confidence_threshold, return_mask)
            
        elif mode == "obj-refine":
            # 如果请求误入此接口，重定向逻辑或直接报错提示
            raise HTTPException(status_code=400, detail="Please use /process-object-refine for this mode")

        # 检查推理结果
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail=f"Inference failed for mode: {mode}")
            
        return FileResponse(output_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(target_path): os.remove(target_path)
        if prompt_path and os.path.exists(prompt_path): os.remove(prompt_path)

@router.post("/process-obj-refine")
async def process_obj_refine_ui(
    target_image: UploadFile = File(...),
    text_prompts: str = Form(""),
    pre_defined_texts: str = Form("person"),
    confidence_threshold: float = Form(0.5),
    return_mask: bool = Form(True),
    merge_results: bool = Form(True),
    crop_config_json: Optional[str] = Form(None)
):
    tmp_path = os.path.join(inference.UPLOADS_DIR, f"refine_{uuid.uuid4()}.jpg")
    try:
        with open(tmp_path, "wb") as b: 
            shutil.copyfileobj(target_image.file, b)
        
        image = cv2.imread(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        refine_texts = [p.strip() for p in text_prompts.split(',') if p.strip()]
        pre_labels = [p.strip() for p in pre_defined_texts.split(',') if p.strip()]

        crop_config = None
        if crop_config_json:
            try:
                crop_config = json.loads(crop_config_json)
            except Exception:
                pass

        output_path = inference.run_obj_refine(tmp_path, refine_texts, pre_labels, confidence_threshold, return_mask, merge_results, crop_config)
        return FileResponse(output_path)

    except Exception as e:
        print(f"Error in process_obj_refine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)