import cv2
import numpy as np
import time
from tokenizers import Tokenizer
import trtsam3  # 导入我们编译好的模块

def osd(image, result):
    for idx, obj in enumerate(results):
        mask = obj.segmentation.mask

        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask # Red channel
        
        # 简单的 alpha blend
        alpha = 0.5
        mask_indices = mask > 0
        image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
        
        # 画框
        x1, y1, x2, y2 = obj.box.left, obj.box.top, obj.box.right, obj.box.bottom
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image

def run():
    # 1. 初始化模型
    print("Initializing engine...")
    engine = trtsam3.Sam3Infer(
        vision_encoder_path="model/vision-encoder.engine",
        text_encoder_path="model/text-encoder.engine",
        decoder_path="model/decoder.engine",
        gpu_id=1,
        confidence_threshold=0.5
    )

    if not engine.load_engines():
        print("Failed to load engines")
        return
    
    tokenizer_path = "tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(length=32, pad_id=49407)
    tokenizer.enable_truncation(max_length=32)

    prompt_text = "helmet"
    encoded = tokenizer.encode(prompt_text)
    input_ids = list(encoded.ids)
    attention_mask = list(encoded.attention_mask)
    
    # 设置 prompt
    engine.setup_text_inputs(prompt_text, input_ids, attention_mask)

    # 3. 读取图像
    image = cv2.imread("images/persons.jpg")
    if image is None:
        print("Image not found")
        return
    for i in range(10):
        engine.forward(image, prompt_text)
    
    start = time.time()
    for i in range(100):
        results = engine.forward(image, prompt_text)
    end = time.time()
    print(f"100 inference times : {(end - start) * 1000} ms")

if __name__ == "__main__":
    run()