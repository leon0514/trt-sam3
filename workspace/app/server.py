import base64
import cv2
import numpy as np
import aiohttp
import time
import sys
import logging  # 仅用于拦截 Uvicorn 日志
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from tokenizers import Tokenizer
import trtsam3
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    拦截标准 logging 日志并转发给 loguru
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 查找调用者的堆栈深度，以便 loguru 显示正确的文件名和行号
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    # 1. 移除 loguru 默认的 handler
    logger.remove()

    # 2. 添加控制台输出
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 3. 添加文件输出 (自动切割 & 异步写入)
    logger.add(
        "logs/service.log",
        rotation="10 MB",      # 每个文件 10MB
        retention="7 days",    # 保留 7 天
        compression="zip",     # 轮转后压缩
        level="INFO",
        enqueue=True,          # 异步写入，避免阻塞主线程
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    # 4. 拦截 Uvicorn 和 FastAPI 的标准日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # 强制将 uvicorn 的日志器重新定向
    for log_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        mod_logger = logging.getLogger(log_name)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False

# 初始化日志配置
setup_logging()

# --- 配置 ---
VISION_ENCODER_PATH = "engine-models/vision-encoder.engine"
TEXT_ENCODER_PATH = "engine-models/text-encoder.engine"
DECODER_PATH = "engine-models/decoder.engine"
GEOMETRY_ENCODER_PATH = "engine-models/geometry-encoder.engine" 
TOKENIZER_PATH = "engine-models/tokenizer.json"
GPU_ID = 0

# 全局变量
engine = None
tokenizer = None

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tokenizer
    try:
        logger.info(f"Initializing models on GPU {GPU_ID}...")
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
        logger.success("Sam3Infer Model Loaded Successfully.")
    except Exception as e:
        # logger.exception 会自动打印堆栈信息
        logger.critical(f"Model init failed: {e}") 
    
    yield
    logger.info("Shutting down service...")

app = FastAPI(title="TendorRT SAM3 Service High-Performance", lifespan=lifespan)

# --- Middleware: 请求日志 ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms
    
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Time={process_time:.2f}ms "
        f"Client={request.client.host}"
    )
    return response

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
    
    confidence_threshold : float = Field(default=0.5, description="object confidence threshold")
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

async def process_image_source(source: InferenceRequest) -> np.ndarray:
    start_decode = time.time()
    img = None
    source_type = "unknown"

    try:
        if source.image_base64:
            source_type = "base64"
            arr = np.frombuffer(base64.b64decode(source.image_base64), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif source.image_url:
            source_type = "url"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(source.image_url) as resp:
                    if resp.status != 200: 
                        logger.error(f"Download failed: {source.image_url} status={resp.status}")
                        raise Exception("Download failed")
                    arr = np.frombuffer(await resp.read(), np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif source.image_path:
            source_type = "path"
            img = cv2.imread(source.image_path)
            if img is None: 
                logger.error(f"Image path invalid: {source.image_path}")
                raise HTTPException(400, "Image path invalid")
        else:
            raise HTTPException(400, "No image source provided")
        
        if img is None:
             raise HTTPException(400, "Image decode result is None")

        decode_cost = (time.time() - start_decode) * 1000
        logger.debug(f"Image loaded via {source_type}. Shape={img.shape}. DecodeTime={decode_cost:.2f}ms") # 改为 debug 减少刷屏
        return img

    except HTTPException:
        raise
    except Exception as e:
        # logger.exception 会自动记录堆栈
        logger.exception(f"Failed to process image source: {e}")
        raise HTTPException(400, f"Image processing failed: {str(e)}")

# --- 接口 ---

@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    if not engine: 
        logger.error("Service unavailable: Engine is None")
        raise HTTPException(503, "Service unavailable")
    if not req.prompts: 
        raise HTTPException(400, "No prompts provided")

    # 1. Decode Image
    img = await process_image_source(req)
    
    t0 = time.time() 

    # 2. Tokenizer Registration
    try:
        unique_texts = set(p.text for p in req.prompts if p.text)
        if unique_texts:
            for text in unique_texts:
                enc = tokenizer.encode(text)
                engine.setup_text_inputs(text, list(enc.ids), list(enc.attention_mask))
    except Exception as e:
        logger.exception(f"Tokenizer error: {e}")
        raise HTTPException(500, "Tokenizer processing failed")

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
        t_infer_start = time.time()
        input_obj = trtsam3.Sam3Input(img, sam3_prompts, confidence_threshold)
        
        batch_results = engine.forwards([input_obj], req.return_mask)
        image_results = batch_results[0]
        t_infer_end = time.time()
    except Exception as e:
        logger.exception(f"Inference critical error: {e}")
        raise HTTPException(500, f"Inference error: {str(e)}")

    # 5. Response Formatting
    dets = []
    try:
        for obj in image_results:
            rle_mask = None
            if req.return_mask and obj.segmentation.mask is not None and obj.segmentation.mask.size > 0:
                rle_mask = binary_mask_to_rle(obj.segmentation.mask)
            
            dets.append(DetectionResult(
                label=obj.class_name,
                score=float(obj.score),
                box=[obj.box.left, obj.box.top, obj.box.right, obj.box.bottom],
                mask=rle_mask
            ))
    except Exception as e:
        logger.exception(f"Result processing error: {e}")
        raise HTTPException(500, "Failed to format results")

    t_total = (time.time() - t0) * 1000
    t_infer = (t_infer_end - t_infer_start) * 1000
    
    logger.info(
        f"Inference Done. "
        f"Prompts={len(req.prompts)} Detections={len(dets)} "
        f"InferTime={t_infer:.2f}ms TotalProcessTime={t_total:.2f}ms"
    )

    return InferenceResponse(results=dets)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)