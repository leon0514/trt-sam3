import os
import shutil
import json
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import inference

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html', media_type='text/html')

@app.post("/process-image")
async def process_image(
    mode: str = Form(...),
    target_image: UploadFile = File(...),
    text_prompts: Optional[str] = Form(None),
    prompt_image: Optional[UploadFile] = File(None),
    target_boxes: Optional[str] = Form(None),
    prompt_boxes: Optional[str] = Form(None),
):
    try:
        target_ext = os.path.splitext(target_image.filename)[1]
        target_filename = f"{uuid.uuid4()}{target_ext}"
        target_save_path = os.path.join(inference.UPLOADS_DIR, target_filename)
        with open(target_save_path, "wb") as buffer:
            shutil.copyfileobj(target_image.file, buffer)

        prompt_save_path = None
        if prompt_image and prompt_image.filename:
            prompt_ext = os.path.splitext(prompt_image.filename)[1]
            prompt_filename = f"{uuid.uuid4()}{prompt_ext}"
            prompt_save_path = os.path.join(inference.UPLOADS_DIR, prompt_filename)
            with open(prompt_save_path, "wb") as buffer:
                shutil.copyfileobj(prompt_image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")

    def parse_boxes(boxes_str: str) -> Optional[list]:
        if not boxes_str: return None
        try:
            return json.loads(boxes_str)
        except (json.JSONDecodeError, KeyError):
            raise HTTPException(status_code=400, detail="无效的框数组格式")

    target_boxes_list = parse_boxes(target_boxes)
    prompt_boxes_list = parse_boxes(prompt_boxes)

    output_path = None
    try:
        if mode == "multi-class":
            prompts = [p.strip() for p in text_prompts.split(',') if p.strip()]
            output_path = inference.run_multi_class_prompt(target_save_path, prompts)
        elif mode == "box":
            output_path = inference.run_box_prompt(target_save_path, target_boxes_list)
        elif mode == "mixed":
            text = text_prompts.split(',')[0].strip()
            output_path = inference.run_mixed_prompt(target_save_path, text, target_boxes_list)
        elif mode == "from-image":
            output_path = inference.run_from_image_prompt(target_save_path, prompt_save_path, prompt_boxes_list)
        else:
            raise HTTPException(status_code=400, detail=f"未知的处理模式: {mode}")

        if not output_path or not os.path.exists(output_path):
             raise HTTPException(status_code=500, detail="模型推理失败，未能生成结果图片。")

        return FileResponse(output_path, media_type="image/jpeg", filename=os.path.basename(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理过程中发生错误: {str(e)}")
    finally:
        if os.path.exists(target_save_path): os.remove(target_save_path)
        if prompt_save_path and os.path.exists(prompt_save_path): os.remove(prompt_save_path)



if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)