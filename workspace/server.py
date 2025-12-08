import base64
import cv2
import numpy as np
import aiohttp
import asyncio
import itertools
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from tokenizers import Tokenizer
import trtsam3

VISION_ENCODER_PATH = "model/vision-encoder.engine"
TEXT_ENCODER_PATH = "model/text-encoder.engine"
DECODER_PATH = "model/decoder.engine"
GEOMETRY_ENCODER_PATH = "" 
TOKENIZER_PATH = "tokenizer.json"
GPU_ID = 0
CONFIDENCE_THRESHOLD = 0.5

app = FastAPI(title="Sam3Infer Service")
engine = None
tokenizer = None

@app.on_event("startup")
def load_model():
    global engine, tokenizer
    try:
        engine = trtsam3.Sam3Infer.create_instance(
            vision_path=VISION_ENCODER_PATH,
            text_path=TEXT_ENCODER_PATH,
            geometry_path=GEOMETRY_ENCODER_PATH,
            decoder_path=DECODER_PATH,
            gpu_id=GPU_ID,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        if engine is None: raise RuntimeError("Engine init failed")
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        tokenizer.enable_padding(length=32, pad_id=49407)
        tokenizer.enable_truncation(max_length=32)
        print("Model Loaded Successfully.")
    except Exception as e:
        print(f"Init failed: {e}")

class ImageSource(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    prompt_text: str

class RLEModel(BaseModel):
    size: List[int]
    counts: List[int]

class DetectionResult(BaseModel):
    left: float
    top: float
    right: float
    bottom: float
    score: float
    mask: Optional[Dict] = None

class InferenceResponse(BaseModel):
    detections: List[DetectionResult]

def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    pixels = (mask > 0).astype(np.uint8).flatten(order='F')
    counts = [len(list(g)) for k, g in itertools.groupby(pixels)]
    if pixels[0] == 1: counts = [0] + counts
    return {'size': list(mask.shape), 'counts': counts}

def rle_to_binary_mask(rle_dict: Dict) -> np.ndarray:
    h, w = rle_dict['size']
    counts = rle_dict['counts']
    mask_flat = np.zeros(h * w, dtype=np.uint8)
    curr, val = 0, 0
    for c in counts:
        if val == 1: mask_flat[curr : curr+c] = 1
        curr += c
        val = 1 - val
    return mask_flat.reshape((h, w), order='F')

async def process_image_source(source: ImageSource) -> np.ndarray:
    if source.image_base64:
        arr = np.frombuffer(base64.b64decode(source.image_base64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if source.image_url:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(source.image_url) as resp:
                return cv2.imdecode(np.frombuffer(await resp.read(), np.uint8), cv2.IMREAD_COLOR)
    if source.image_path:
        return cv2.imread(source.image_path)
    raise HTTPException(400, "No image source provided")

@app.post("/predict", response_model=InferenceResponse)
async def predict(source: ImageSource, return_mask: bool = False):
    if not engine: raise HTTPException(503, "Service not ready")
    img = await process_image_source(source)
    
    enc = tokenizer.encode(source.prompt_text)
    engine.setup_text_inputs(source.prompt_text, list(enc.ids), list(enc.attention_mask))
    
    try:
        input_obj = trtsam3.Sam3Input(img, source.prompt_text)
        results = engine.forward(input_obj)
    except Exception as e:
        raise HTTPException(500, str(e))

    dets = []
    for obj in results:
        rle = None
        if return_mask and obj.segmentation.mask is not None:
            rle = binary_mask_to_rle(obj.segmentation.mask)
        dets.append(DetectionResult(
            left=obj.box.left, top=obj.box.top, right=obj.box.right, bottom=obj.box.bottom,
            score=obj.score, mask=rle
        ))
    return InferenceResponse(detections=dets)

# @app.post("/visualize-mask", responses={200: {"content": {"image/png": {}}}})
# async def visualize_mask(rle: RLEModel):
#     try:
#         mask = rle_to_binary_mask({"size": rle.size, "counts": rle.counts})
#         vis = (mask * 255).astype(np.uint8)
#         ret, buf = cv2.imencode(".png", vis)
#         return Response(content=buf.tobytes(), media_type="image/png")
#     except Exception as e:
#         raise HTTPException(400, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)