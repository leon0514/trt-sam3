import cv2
import numpy as np
import os
import random
from tokenizers import Tokenizer
import trtsam3  

# --- 配置路径 ---
VISION_MODEL = "model/vision-encoder.engine"
TEXT_MODEL = "model/text-encoder.engine"
DECODER_MODEL = "model/decoder.engine"
GEOMETRY_MODEL = "model/geometry-encoder.engine" # 如果需要框/点提示，必须加载
TOKENIZER_PATH = "tokenizer.json"
OUTPUT_DIR = "output"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_random_color(seed_str):
    """根据字符串生成固定颜色"""
    random.seed(hash(seed_str))
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def osd(image, results):
    """
    可视化函数：绘制 Mask, Box 和 Label
    """
    if not results: 
        return image
    
    vis_img = image.copy()
    
    for obj in results:
        # 获取颜色 (根据类别名)
        color = get_random_color(obj.class_name)
        
        # 1. 绘制 Mask (半透明)
        mask = obj.segmentation.mask
        if mask is not None and mask.size > 0:
            colored_mask = np.zeros_like(vis_img)
            colored_mask[:, :, 0] = color[0]
            colored_mask[:, :, 1] = color[1]
            colored_mask[:, :, 2] = color[2]
            
            # 仅在 mask 区域混合颜色
            mask_indices = mask > 0
            vis_img[mask_indices] = cv2.addWeighted(
                vis_img[mask_indices], 0.5, 
                colored_mask[mask_indices], 0.5, 0
            )
        
        # 2. 绘制 Box
        x1, y1, x2, y2 = int(obj.box.left), int(obj.box.top), int(obj.box.right), int(obj.box.bottom)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 3. 绘制标签和置信度
        label_text = f"{obj.class_name}: {obj.score:.2f}"
        cv2.putText(vis_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return vis_img

def init_system(gpu_id=0):
    print(f"Initializing engine on GPU {gpu_id}...")
    
    # 1. 创建推理引擎
    engine = trtsam3.Sam3Infer.create_instance(
        vision_path=VISION_MODEL,
        text_path=TEXT_MODEL,
        geometry_path=GEOMETRY_MODEL, 
        decoder_path=DECODER_MODEL,
        gpu_id=gpu_id,
        confidence_threshold=0.4
    )
    
    if engine is None:
        raise RuntimeError("Failed to load TensorRT engines.")

    # 2. 加载 Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
        
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_padding(length=32, pad_id=49407)
    tokenizer.enable_truncation(max_length=32)
    
    return engine, tokenizer

def register_prompts(engine, tokenizer, text_list):
    """
    辅助函数：将文本 Token 化并注册到 C++ 引擎中。
    """
    print(f"Registering tokens for: {text_list}")
    for text in text_list:
        if not text: continue
        encoded = tokenizer.encode(text)
        engine.setup_text_inputs(text, encoded.ids, encoded.attention_mask)

# ==============================================================================
# 场景 1: 框提示 (Box Prompt) 测试
# ==============================================================================
def demo_box_prompt(engine):
    print("\n=== Running Box Prompt Test ===")
    image_path = "images/persons.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping: {image_path} not found.")
        return

    # 定义一个框 (格式: C++ std::pair<string, array<float,4>>)
    # Python 对应: ("pos", [x1, y1, x2, y2])
    # 注意：坐标需根据实际图片调整
    box_prompt = ("pos", [747, 319, 768, 384]) 
    
    prompt_unit = trtsam3.Sam3PromptUnit("", [box_prompt])
    input_obj = trtsam3.Sam3Input(image, [prompt_unit])

    # 推理
    results = engine.forwards([input_obj])[0] # 获取第一张图的结果
    
    # 可视化
    # 画出提示框 (蓝色) 以便对比
    bx1, by1, bx2, by2 = map(int, box_prompt[1])
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    cv2.putText(image, "Prompt Box", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    vis_img = osd(image, results)
    
    save_path = os.path.join(OUTPUT_DIR, "demo_box.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"Result saved to {save_path}")

# ==============================================================================
# 场景 2: 单图多文本 Prompt (One Vision, Multiple Prompts)
# 这是本次 C++ 代码优化的核心体现
# ==============================================================================
def demo_multi_class_prompt(engine, tokenizer):
    print("\n=== Running Multi-Class Prompt Test ===")
    image_path = "images/persons.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping: {image_path} not found.")
        return

    # 我们想要在这张图中同时识别 "person" 和 "glasses" (或者其他物体)
    prompts_text = ["tie", "glasses", "head"]
    
    # 1. 注册 Token (重要)
    register_prompts(engine, tokenizer, prompts_text)

    # 2. 构造 Prompt 列表
    # C++ 内部会将这 3 个 Prompt 复用同一份 Vision Feature
    prompt_units = []
    for txt in prompts_text:
        prompt_units.append(trtsam3.Sam3PromptUnit(txt))
    
    # 3. 构造 Input 对象
    input_obj = trtsam3.Sam3Input(image, prompt_units)

    # 4. 推理
    # engine.forwards 接受一个 inputs 列表，这里我们只传一张图
    batch_results = engine.forwards([input_obj])
    
    # 5. 获取结果
    # batch_results[0] 包含了该张图片下所有 Prompt 检测到的物体
    image_results = batch_results[0]
    
    print(f"Detected {len(image_results)} objects across {len(prompts_text)} prompts.")
    for obj in image_results:
        print(f" - Found {obj.class_name} with score {obj.score:.3f}")

    # 6. 可视化
    vis_img = osd(image, image_results)
    
    save_path = os.path.join(OUTPUT_DIR, "demo_multi_class.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"Result saved to {save_path}")

# ==============================================================================
# 场景 3: 混合 Prompt (文本 + 框)
# ==============================================================================
def demo_mixed_prompt(engine, tokenizer):
    print("\n=== Running Mixed Prompt Test ===")
    image_path = "images/persons.jpg"
    image = cv2.imread(image_path)
    if image is None: return

    target_text = "tie"
    register_prompts(engine, tokenizer, [target_text])

    box_constraint = ("pos", [747, 319, 768, 384])
    
    prompt_unit = trtsam3.Sam3PromptUnit(target_text, [box_constraint])
    input_obj = trtsam3.Sam3Input(image, [prompt_unit])
    
    results = engine.forwards([input_obj])[0]
    
    # 可视化提示框
    bx1, by1, bx2, by2 = map(int, box_constraint[1])
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
    
    vis_img = osd(image, results)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "demo_mixed.jpg"), vis_img)
    print(f"Result saved to {os.path.join(OUTPUT_DIR, 'demo_mixed.jpg')}")

if __name__ == "__main__":
    try:
        # 初始化资源
        engine, tokenizer = init_system(gpu_id=0)
        
        # 运行不同的 Demo
        demo_multi_class_prompt(engine, tokenizer) # 核心新功能
        demo_box_prompt(engine)                    # 纯几何提示
        demo_mixed_prompt(engine, tokenizer)       # 混合提示
        
        print("\nAll demos finished successfully.")
        
    except Exception as e:
        print(f"\nError occurred: {e}")