import base64
import cv2
import numpy as np
import aiohttp
import asyncio
import itertools
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from tokenizers import Tokenizer
import trtsam3


VISION_ENCODER_PATH = "model/vision-encoder.engine"
TEXT_ENCODER_PATH = "model/text-encoder.engine"
DECODER_PATH = "model/decoder.engine"
TOKENIZER_PATH = "tokenizer.json"
GPU_ID = 0
CONFIDENCE_THRESHOLD = 0.5

app = FastAPI(
    title="高效图像识别服务",
    description="一个基于 trtsam3 的 FastAPI 服务，返回紧凑的 JSON 结果。"
)
engine = None
tokenizer = None

@app.on_event("startup")
def load_model():
    """
    在 FastAPI 应用启动时加载和初始化所有模型和分词器。
    """
    global engine, tokenizer
    print("正在初始化推理引擎...")
    try:
        engine = trtsam3.Sam3Infer(
            vision_encoder_path=VISION_ENCODER_PATH,
            text_encoder_path=TEXT_ENCODER_PATH,
            decoder_path=DECODER_PATH,
            gpu_id=GPU_ID,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        if not engine.load_engines():
            raise RuntimeError("加载 TensorRT 引擎失败")
        
        print("推理引擎初始化成功。")

        print("正在加载分词器...")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        tokenizer.enable_padding(length=32, pad_id=49407)
        tokenizer.enable_truncation(max_length=32)
        print("分词器加载成功。")

    except Exception as e:
        print(f"初始化过程中发生错误: {e}")
        engine = None
        tokenizer = None

class ImageSource(BaseModel):
    image_base64: Optional[str] = Field(None, description="图片的 Base64 编码字符串")
    image_url: Optional[str] = Field(None, description="图片的公开可访问 URL")
    image_path: Optional[str] = Field(None, description="图片在服务器上的本地路径")
    prompt_text: str = Field(..., description="用于识别的文本提示，例如 'helmet' 或 'coal'")

class DetectionResult(BaseModel):
    left: float = Field(..., description="边界框左上角 x 坐标")
    top: float = Field(..., description="边界框左上角 y 坐标")
    right: float = Field(..., description="边界框右下角 x 坐标")
    bottom: float = Field(..., description="边界框右下角 y 坐标")
    score: float = Field(..., description="检测结果的置信度")
    mask: Optional[Dict] = Field(None, description="分割掩码的 RLE (Run-Length Encoding) 编码")

class InferenceResponse(BaseModel):
    detections: List[DetectionResult]


# ------------------- 辅助函数 -------------------
def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    将二值掩码转换为 COCO 风格的 RLE (Run-Length Encoding) 格式。

    Args:
        mask: 一个二维的 numpy 数组，其中 >0 的值被视为掩码部分。

    Returns:
        一个包含 'size' 和 'counts' 的字典，代表 RLE 编码。
    """
    # 确保掩码是二值的 (0 和 1)
    pixels = (mask > 0).astype(np.uint8)
    
    # RLE 编码在列优先（Fortran-style）展开的数组上进行
    pixels = pixels.flatten(order='F')
    
    # 使用 itertools.groupby 来获取连续相同值的长度
    counts = [len(list(g)) for k, g in itertools.groupby(pixels)]
    
    # RLE 格式要求以 0 的计数开始。如果我们的掩码以 1 开始，c

    # 我们需要在计数的开头插入一个 0。
    if pixels[0] == 1:
        counts = [0] + counts
        
    return {'size': list(mask.shape), 'counts': counts}

async def process_image_source(source: ImageSource) -> np.ndarray:
    """
    根据输入源（Base64, URL, Path）解码或下载图片，并返回 OpenCV 格式的图像。
    """
    if source.image_base64:
        try:
            img_data = base64.b64decode(source.image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None: raise ValueError("无法解码 Base64 图片")
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的 Base64 字符串: {e}")

    if source.image_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source.image_url) as response:
                    response.raise_for_status()
                    img_data = await response.read()
                    np_arr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if image is None: raise ValueError("下载的内容不是有效的图片格式")
                    return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法从 URL 下载或解码图片: {e}")

    if source.image_path:
        try:
            image = cv2.imread(source.image_path)
            if image is None: raise FileNotFoundError(f"在路径 '{source.image_path}' 未找到图片")
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法读取本地图片文件: {e}")
    
    raise HTTPException(status_code=400, detail="必须提供 'image_base64', 'image_url', 或 'image_path' 中的一种")


@app.post("/predict", response_model=InferenceResponse)
async def predict(source: ImageSource, return_mask: bool = False):
    """
    接收图片和文本提示，执行识别并返回 JSON 格式的结果。

    - **source**: 包含图片来源和文本提示的对象。
    - **return_mask**: 是否在返回的 JSON 中包含每个目标的 RLE 编码掩码。
    """
    if not engine or not tokenizer:
        raise HTTPException(status_code=503, detail="服务尚未准备好，模型正在加载或加载失败。")

    image = await process_image_source(source)
    
    try:
        encoded = tokenizer.encode(source.prompt_text)
        engine.setup_text_inputs(source.prompt_text, list(encoded.ids), list(encoded.attention_mask))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文本提示时出错: {e}")

    try:
        results = engine.forward(image, source.prompt_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理过程中发生错误: {e}")

    detections = []
    for obj in results:
        rle_mask = None
        if return_mask and hasattr(obj, 'segmentation') and hasattr(obj.segmentation, 'mask'):
            rle_mask = binary_mask_to_rle(obj.segmentation.mask)

        detection = DetectionResult(
            left=obj.box.left,
            top=obj.box.top,
            right=obj.box.right,
            bottom=obj.box.bottom,
            score=obj.score, 
            mask=rle_mask
        )
        detections.append(detection)
    
    return InferenceResponse(detections=detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)