import base64
import cv2
import numpy as np
import aiohttp
import itertools
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from tokenizers import Tokenizer
import trtsam3

# --- 配置 ---
VISION_ENCODER_PATH = "model/vision-encoder.engine"
TEXT_ENCODER_PATH = "model/text-encoder.engine"
DECODER_PATH = "model/decoder.engine"
GEOMETRY_ENCODER_PATH = "model/geometry-encoder.engine" 
TOKENIZER_PATH = "tokenizer.json"
GPU_ID = 0

# 全局变量
engine = None
tokenizer = None

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tokenizer
    try:
        print(f"Loading models on GPU {GPU_ID}...")
        engine = trtsam3.Sam3Infer.create_instance(
            vision_path=VISION_ENCODER_PATH,
            text_path=TEXT_ENCODER_PATH,
            geometry_path=GEOMETRY_ENCODER_PATH,
            decoder_path=DECODER_PATH,
            gpu_id=GPU_ID
        )
        if engine is None: 
            raise RuntimeError("Engine init failed (returned None)")
            
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        tokenizer.enable_padding(length=32, pad_id=49407)
        tokenizer.enable_truncation(max_length=32)
        print("Sam3Infer Model Loaded Successfully.")
    except Exception as e:
        print(f"CRITICAL: Model init failed: {e}")
    
    yield
    print("Shutting down...")

app = FastAPI(title="Sam3Infer Service High-Performance", lifespan=lifespan)

# --- 数据模型 ---

class BoxInput(BaseModel):
    label: str = Field(default="pos", description="'pos' for positive, 'neg' for negative")
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates", min_length=4, max_length=4)

class PromptUnit(BaseModel):
    text: Optional[str] = Field(default="", description="Text prompt")
    boxes: Optional[List[BoxInput]] = Field(default=[], description="Geometric boxes")

class InferenceRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    
    confidence_threshold : float = Field(default=0.65, description="object confidence threshold")
    prompts: List[PromptUnit] = Field(..., description="List of prompts")
    return_mask: bool = Field(default=False, description="If True, returns segmentation masks")

class DetectionResult(BaseModel):
    label: str
    score: float
    box: List[float]
    mask: Optional[Dict] = None

class InferenceResponse(BaseModel):
    results: List[DetectionResult]

# --- 核心优化函数 ---

def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    使用 Numpy 向量化加速 RLE 编码。
    比 itertools.groupby 快 50-100 倍。
    """
    if mask is None: return None
    
    # 1. 扁平化 (Fortran-style / 列优先，符合 COCO RLE 标准)
    # 使用 ravel 尝试避免复制，astype 确保是二值
    flat = mask.ravel(order='F')
    flat = (flat > 0).astype(np.int8)
    
    if len(flat) == 0:
        return {'size': list(mask.shape), 'counts': []}

    # 2. 寻找跳变点 (Vectorized)
    # flat[1:] != flat[:-1] 得到布尔数组，np.where 得到索引
    # 相比 Python 循环，这里是 C 语言级别的速度
    diffs = np.where(flat[1:] != flat[:-1])[0] + 1
    
    # 3. 计算 Run Lengths
    # 在开头补0，结尾补总长度，做差分即可得到每段长度
    # indices: [0, idx1, idx2, ..., total_len]
    bounds = np.concatenate(([0], diffs, [len(flat)]))
    counts = np.diff(bounds)
    
    # 4. 转换为 List (JSON序列化需要 int，而不是 numpy.int)
    counts_list = counts.tolist()
    
    # 5. 处理 RLE 规范：第一个数必须是背景(0)的数量
    # 如果 flat[0] 是 1 (前景)，说明开头有 0 个背景
    if flat[0] == 1:
        counts_list = [0] + counts_list
        
    return {'size': list(mask.shape), 'counts': counts_list}

async def process_image_source(source: InferenceRequest) -> np.ndarray:
    if source.image_base64:
        try:
            arr = np.frombuffer(base64.b64decode(source.image_base64), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            raise HTTPException(400, "Invalid Base64 image")
    if source.image_url:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(source.image_url) as resp:
                    if resp.status != 200: raise Exception("Download failed")
                    return cv2.imdecode(np.frombuffer(await resp.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(400, f"Failed fetch URL: {str(e)}")
    if source.image_path:
        img = cv2.imread(source.image_path)
        if img is None: raise HTTPException(400, "Image path invalid")
        return img
    raise HTTPException(400, "No image source provided")

# --- 接口 ---

@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    if not engine: raise HTTPException(503, "Service unavailable")
    if not req.prompts: raise HTTPException(400, "No prompts provided")

    # 1. Decode Image
    img = await process_image_source(req)
    if img is None: raise HTTPException(400, "Image decode failed")

    # 2. Tokenizer Registration
    unique_texts = set(p.text for p in req.prompts if p.text)
    if unique_texts:
        for text in unique_texts:
            enc = tokenizer.encode(text)
            engine.setup_text_inputs(text, list(enc.ids), list(enc.attention_mask))

    # 3. Construct Input
    sam3_prompts = []
    for p in req.prompts:
        cpp_boxes = []
        if p.boxes:
            for b in p.boxes:
                if len(b.bbox) == 4: cpp_boxes.append((b.label, b.bbox))
        sam3_prompts.append(trtsam3.Sam3PromptUnit(p.text, cpp_boxes))

    confidence_threshold = req.confidence_threshold
    # 4. Inference
    try:
        input_obj = trtsam3.Sam3Input(img, sam3_prompts, confidence_threshold)
        
        # 传入 req.return_mask 控制是否解码 Mask
        batch_results = engine.forwards([input_obj], req.return_mask)
        
        image_results = batch_results[0]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Inference error: {str(e)}")

    import time
    s = time.time()
    # 5. Response
    dets = []
    for obj in image_results:
        rle_mask = None
        # 如果需要 mask 且 mask 存在
        if req.return_mask and obj.segmentation.mask is not None and obj.segmentation.mask.size > 0:
            # 使用加速后的函数
            rle_mask = binary_mask_to_rle(obj.segmentation.mask)
        
        dets.append(DetectionResult(
            label=obj.class_name,
            score=float(obj.score),
            box=[obj.box.left, obj.box.top, obj.box.right, obj.box.bottom],
            mask=rle_mask
        ))
    return InferenceResponse(results=dets)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)