import cv2
import numpy as np
import os
import random
import uuid
import time
from tokenizers import Tokenizer
import trtsam3
from typing import List, Any, Tuple, Optional
from utils import merge_person_boxes, get_person_regions, nms


# --- 配置路径 ---
VISION_MODEL = "../engine-models/vision-encoder.engine"
TEXT_MODEL = "../engine-models/text-encoder.engine"
DECODER_MODEL = "../engine-models/decoder.engine"
GEOMETRY_MODEL = "../engine-models/geometry-encoder.engine"
TOKENIZER_PATH = "../engine-models/tokenizer.json"
OUTPUT_DIR = "outputs"
UPLOADS_DIR = "uploads"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

class ModelManager:
    """管理模型生命周期，确保只加载一次"""
    engine: trtsam3.Sam3Infer = None
    tokenizer: Tokenizer = None

    @classmethod
    def initialize(cls, gpu_id: int = 0):
        if cls.engine is not None:
            return
        print(f"Initializing engine on GPU {gpu_id}...")
        cls.engine = trtsam3.Sam3Infer.create_instance(
            vision_path=VISION_MODEL, text_path=TEXT_MODEL,
            geometry_path=GEOMETRY_MODEL, decoder_path=DECODER_MODEL, gpu_id=gpu_id
        )
        if cls.engine is None:
            raise RuntimeError("Failed to load TensorRT engines.")
        
        cls.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        cls.tokenizer.enable_padding(length=32, pad_id=49407)
        cls.tokenizer.enable_truncation(max_length=32)
        print("Models loaded successfully!")

def register_prompts(text_list: List[str]):
    """注册文本提示词"""
    for text in text_list:
        if not text: continue
        encoded = ModelManager.tokenizer.encode(text)
        ModelManager.engine.setup_text_inputs(text, encoded.ids, encoded.attention_mask)

# --- 核心推理函数 ---

def inference_with_box_prompt(image: np.ndarray, boxes: List[dict], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None or not boxes: return None
    # 格式转换适配底层
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit("", prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]

def inference_with_multi_class_prompt(image: np.ndarray, prompts: List[str], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None: return None
    register_prompts(prompts)
    prompt_units = [trtsam3.Sam3PromptUnit(txt) for txt in prompts]
    input_obj = trtsam3.Sam3Input(image, prompt_units, threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]

def inference_with_mixed_prompt(image: np.ndarray, text_prompt: str, boxes: List[dict], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None: return None
    if text_prompt: register_prompts([text_prompt])
    # 适配转换
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit(text_prompt, prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]

def setup_image_prompt(prompt_image: np.ndarray, boxes: List[dict]) -> Optional[str]:
    if prompt_image is None or not boxes: return None
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    geom_label = f"dynamic_prompt_{uuid.uuid4()}"
    ok = ModelManager.engine.setup_geometry_input(prompt_image, geom_label, prompts)
    return geom_label if ok else None

def inference_with_image_prompt(target_image: np.ndarray, geom_label: str, threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if target_image is None or not geom_label: return None
    input_obj = trtsam3.Sam3Input(target_image, [], threshold)
    return ModelManager.engine.forwards([input_obj], geom_label, return_results)[0]


def inference_with_obj_refine(
    image: np.ndarray,
    refine_texts: List[str],
    pre_defined_texts: List[str] = ["person"],
    threshold: float = 0.5,
    return_mask: bool = True,
    merge_results: bool = True,
    crop_config: dict = None
) -> Optional[List[Any]]:
    if image is None: return None

    t0 = time.time()

    # 注册所有需要的 token（预检测标签 + 精细检测标签）
    all_texts = list(set(pre_defined_texts + refine_texts))
    register_prompts(all_texts)

    # 构造精细检测 prompts
    prompt_units = [trtsam3.Sam3PromptUnit(txt) for txt in refine_texts]

    # 构造 Sam3Input，使用 C++ 层的 pre_detect_labels + merge_results
    input_obj = trtsam3.Sam3Input(image, prompt_units, threshold)
    input_obj.pre_detect_labels = pre_defined_texts
    input_obj.merge_results = merge_results

    # 应用 ominicrop 配置
    if crop_config:
        input_obj.pre_crop_max_size = crop_config.get('max_size', 640)
        input_obj.pre_crop_padding = crop_config.get('padding', 20)
        input_obj.pre_crop_w_diou = crop_config.get('w_diou', 30.0)
        input_obj.pre_crop_w_expansion = crop_config.get('w_expansion', 5.0)
        input_obj.pre_crop_count_penalty = crop_config.get('count_penalty', 120.0)
        input_obj.pre_crop_nms_threshold = crop_config.get('nms_threshold', 0.2)
        input_obj.pre_crop_enable_ar_fix = crop_config.get('enable_ar_fix', True)
        input_obj.pre_crop_target_ar = crop_config.get('target_ar', 1.0)

    results = ModelManager.engine.forwards([input_obj], return_mask)[0]

    t1 = time.time()
    print(f"[inference_with_obj_refine] pre_detect={pre_defined_texts}, refine={refine_texts}, "
          f"merge={merge_results}, results={len(results)}, time={t1-t0:.3f}s")

    return results

def draw_and_save_image(image: np.ndarray, results: List[Any], prefix: str = "result") -> str:
    output_image = image.copy()
    if results and len(results) > 0:
        trtsam3.osd(output_image, results, True, 0.04)
    save_path = os.path.join(OUTPUT_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(save_path, output_image)
    return save_path


def run_box_prompt(image_path: str, boxes: List[dict], confidence_threshold: float = 0.5, return_results: bool = True) -> Optional[str]:
    image = cv2.imread(image_path)
    if image is None: return None
    results = inference_with_box_prompt(image, boxes, confidence_threshold, return_results)
    return draw_and_save_image(image, results, "box_prompt")

def run_multi_class_prompt(image_path: str, prompts: List[str], confidence_threshold: float = 0.5, return_results: bool = True) -> Optional[str]:
    image = cv2.imread(image_path)
    if image is None: return None
    results = inference_with_multi_class_prompt(image, prompts, confidence_threshold, return_results)
    return draw_and_save_image(image, results, "multi_class_prompt")

def run_mixed_prompt(image_path: str, text_prompt: str, boxes: List[dict], confidence_threshold: float = 0.5, return_results: bool = True) -> Optional[str]:
    image = cv2.imread(image_path)
    if image is None: return None
    results = inference_with_mixed_prompt(image, text_prompt, boxes, confidence_threshold, return_results)
    return draw_and_save_image(image, results, "mixed_prompt")

def run_from_image_prompt(target_image_path: str, prompt_image_path: str, boxes: List[dict], confidence_threshold: float = 0.5, return_results: bool = True) -> Optional[str]:
    prompt_img = cv2.imread(prompt_image_path)
    target_img = cv2.imread(target_image_path)
    geom_label = setup_image_prompt(prompt_img, boxes)
    if not geom_label: return None
    results = inference_with_image_prompt(target_img, geom_label, confidence_threshold, return_results)
    return draw_and_save_image(target_img, results, "image_prompt")

def run_obj_refine(
    image_path: str,
    refine_texts: List[str],
    pre_defined_texts: List[str] = ["person"],
    confidence_threshold: float = 0.5,
    return_mask: bool = True,
    merge_results: bool = True,
    crop_config: dict = None
) -> Optional[str]:
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None: return None
    results = inference_with_obj_refine(image, refine_texts, pre_defined_texts, confidence_threshold, return_mask, merge_results, crop_config)
    save_path = draw_and_save_image(image, results, "obj_refine")
    t1 = time.time()
    print(f"[run_obj_refine] total_time={t1-t0:.3f}s, save_path={save_path}")
    return save_path
