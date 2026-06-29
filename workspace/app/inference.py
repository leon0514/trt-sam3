import cv2
import numpy as np
import uuid
import time
from tokenizers import Tokenizer
import trtsam3
from typing import List, Any, Optional


# --- 配置路径 ---
VISION_MODEL = "engine-models/vision-encoder.engine"
TEXT_MODEL = "engine-models/text-encoder.engine"
DECODER_MODEL = "engine-models/decoder.engine"
GEOMETRY_MODEL = "engine-models/geometry-encoder.engine"  # 空字符串表示没有单独的几何编码器
TOKENIZER_PATH = "engine-models/tokenizer.json"


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
        if not text:
            continue
        encoded = ModelManager.tokenizer.encode(text)
        ModelManager.engine.setup_text_inputs(text, encoded.ids, encoded.attention_mask)


# --- 核心推理函数 ---

def inference_with_box_prompt(image: np.ndarray, boxes: List[dict], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None or not boxes:
        return None
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit("", prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]


def inference_with_multi_class_prompt(image: np.ndarray, prompts: List[str], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None:
        return None
    register_prompts(prompts)
    prompt_units = [trtsam3.Sam3PromptUnit(txt) for txt in prompts]
    input_obj = trtsam3.Sam3Input(image, prompt_units, threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]


def inference_with_mixed_prompt(image: np.ndarray, text_prompt: str, boxes: List[dict], threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if image is None:
        return None
    if text_prompt:
        register_prompts([text_prompt])
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    prompt_unit = trtsam3.Sam3PromptUnit(text_prompt, prompts)
    input_obj = trtsam3.Sam3Input(image, [prompt_unit], threshold)
    return ModelManager.engine.forwards([input_obj], return_results)[0]


def setup_image_prompt(prompt_image: np.ndarray, boxes: List[dict]) -> Optional[str]:
    if prompt_image is None or not boxes:
        return None
    prompts = [(box['type'], [box['x1'], box['y1'], box['x2'], box['y2']]) for box in boxes]
    geom_label = f"dynamic_prompt_{uuid.uuid4()}"
    ok = ModelManager.engine.setup_geometry_input(prompt_image, geom_label, prompts)
    return geom_label if ok else None


def inference_with_image_prompt(target_image: np.ndarray, geom_label: str, threshold: float = 0.5, return_results: bool = True) -> Optional[List[Any]]:
    if target_image is None or not geom_label:
        return None
    input_obj = trtsam3.Sam3Input(target_image, [], threshold)
    return ModelManager.engine.forwards([input_obj], geom_label, return_results)[0]


def inference_with_obj_refine(
    image: np.ndarray,
    refine_texts: List[str],
    pre_defined_texts: List[str] = None,
    threshold: float = 0.5,
    return_mask: bool = True,
    merge_results: bool = True,
    crop_config: dict = None
) -> Optional[List[Any]]:
    if image is None:
        return None

    if pre_defined_texts is None:
        pre_defined_texts = ["person"]

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
