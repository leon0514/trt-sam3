import cv2
import numpy as np
import os
import random
import uuid
from tokenizers import Tokenizer
import trtsam3

# --- 配置路径 ---
# VISION_MODEL = "engine-models/vision-encoder-int8.engine"
VISION_MODEL = "engine-models/vision-encoder.engine"
TEXT_MODEL = "engine-models/text-encoder.engine"
DECODER_MODEL = "engine-models/decoder.engine"
GEOMETRY_MODEL = "engine-models/geometry-encoder.engine"
TOKENIZER_PATH = "engine-models/tokenizer.json"
OUTPUT_DIR = "outputs"
UPLOADS_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- 辅助函数 ---
def get_random_color(seed_str):
    random.seed(hash(seed_str))
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def init_system(gpu_id=0):
    print(f"Initializing engine on GPU {gpu_id}...")
    engine = trtsam3.Sam3Infer.create_instance(
        vision_path=VISION_MODEL, text_path=TEXT_MODEL,
        geometry_path=GEOMETRY_MODEL, decoder_path=DECODER_MODEL, gpu_id=gpu_id
    )
    if engine is None: raise RuntimeError("Failed to load TensorRT engines.")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_padding(length=32, pad_id=49407)
    tokenizer.enable_truncation(max_length=32)
    return engine, tokenizer

print("Loading models, please wait...")
engine, tokenizer = init_system(1)
print("Models loaded successfully!")

def register_prompts(text_list):
    print(f"Registering tokens for: {text_list}")
    for text in text_list:
        if not text: continue
        encoded = tokenizer.encode(text)
        engine.setup_text_inputs(text, encoded.ids, encoded.attention_mask)

def _save_and_return_image(image, prefix="result"):
    save_path = os.path.join(OUTPUT_DIR, f"{prefix}.jpg")
    cv2.imwrite(save_path, image)
    return save_path

def run_box_prompt(image_path: str, boxes: list) -> str:
    image = cv2.imread(image_path)
    if image is None or not boxes: return None
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit("", prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], 0.5)
    results = engine.forwards([input_obj], True)[0]
    trtsam3.osd(image, results, True, 0.04)
    return _save_and_return_image(image)

def run_multi_class_prompt(image_path: str, prompts: list) -> str:
    image = cv2.imread(image_path)
    if image is None: return None
    register_prompts(prompts)
    prompt_units = [trtsam3.Sam3PromptUnit(txt) for txt in prompts]
    input_obj = trtsam3.Sam3Input(image, prompt_units, 0.4)
    image_results = engine.forwards([input_obj], True)[0]
    trtsam3.osd(image, image_results, True, 0.04)
    return _save_and_return_image(image)

def run_mixed_prompt(image_path: str, text_prompt: str, boxes: list) -> str:
    image = cv2.imread(image_path)
    if image is None or not boxes: return None
    register_prompts([text_prompt])
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit(text_prompt, prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], 0.5)
    results = engine.forwards([input_obj], True)[0]
    trtsam3.osd(image, results, True, 0.04)
    return _save_and_return_image(image)

def run_from_image_prompt(target_image_path: str, prompt_image_path: str, boxes: list) -> str:
    prompt_image = cv2.imread(prompt_image_path)
    target_image = cv2.imread(target_image_path)
    if prompt_image is None or target_image is None or not boxes: return None
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    geom_label = f"dynamic_prompt_{uuid.uuid4()}"
    ok = engine.setup_geometry_input(prompt_image, geom_label, prompts)
    if not ok: return None
    input_obj = trtsam3.Sam3Input(target_image, [], 0.5)
    results = engine.forwards([input_obj], geom_label, True)[0]
    trtsam3.osd(target_image, results, True, 0.04)
    return _save_and_return_image(target_image)