import cv2
import numpy as np
import time
from tokenizers import Tokenizer
import trtsam3  


def osd(image, results):
    if not results: return image
    for idx, obj in enumerate(results):
        mask = obj.segmentation.mask
        if mask is None or mask.size == 0: continue

        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 2] = mask # Red channel
        
        alpha = 0.5
        mask_indices = mask > 0
        image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
        
        x1, y1, x2, y2 = obj.box.left, obj.box.top, obj.box.right, obj.box.bottom
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return image

def get_engine_and_tokenizer(gpu_id=0):
    print(f"Initializing engine on GPU {gpu_id}...")
    engine = trtsam3.Sam3Infer.create_instance(
        vision_path="model/vision-encoder.engine",
        text_path="model/text-encoder.engine",
        geometry_path="",
        decoder_path="model/decoder.engine",
        gpu_id=gpu_id,
        confidence_threshold=0.5
    )
    
    if engine is None:
        raise RuntimeError("Failed to load engine")

    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer.enable_padding(length=32, pad_id=49407)
    tokenizer.enable_truncation(max_length=32)
    
    return engine, tokenizer

def speed_test():
    engine, tokenizer = get_engine_and_tokenizer(gpu_id=1)

    prompt_text = "helmet"
    encoded = tokenizer.encode(prompt_text)
    engine.setup_text_inputs(prompt_text, encoded.ids, encoded.attention_mask)

    image = cv2.imread("images/persons.jpg")
    if image is None:
        print("Image not found")
        return
    
    # 构造 Input 对象
    input_obj = trtsam3.Sam3Input(image, prompt_text)
    
    print("Warm up ...")
    for i in range(5):
        engine.forward(input_obj)
    
    print("Benchmarking 100 iterations...")
    start = time.time()
    for i in range(100):
        # 仍然使用 forward 接口，但传入 Input 对象
        engine.forward(input_obj)
    end = time.time()
    
    avg_time = (end - start) * 1000 / 100
    fps = 1000 / avg_time
    print(f"Avg Latency: {avg_time:.2f} ms, FPS: {fps:.2f}")

    # 测试 Batch Forward (Sam3Input 列表)
    batch_inputs = [input_obj, input_obj, input_obj, input_obj] # Batch=4
    print("\nBenchmarking Batch=4 ...")
    start = time.time()
    for i in range(25): # 100 images total
        engine.forwards(batch_inputs)
    end = time.time()
    avg_time_batch = (end - start) * 1000 / 25
    print(f"Batch(4) Latency: {avg_time_batch:.2f} ms, FPS: {1000/(avg_time_batch/4):.2f}")


def run_box_prompt_test():
    print("\n=== Running Box Prompt Test ===")
    
    # 1. 初始化引擎 (必须包含 geometry_path)
    engine = trtsam3.Sam3Infer.create_instance(
        vision_path="model/vision-encoder.engine",
        text_path="model/text-encoder.engine",
        geometry_path="model/geometry-encoder.engine", # 关键：加载 Geometry
        decoder_path="model/decoder.engine",
        gpu_id=0,
        confidence_threshold=0.5
    )
    
    if engine is None:
        print("Failed to load engine")
        return

    # 2. 读取图像
    image = cv2.imread("images/smx.jpg")
    if image is None:
        print("Image not found")
        return

    box_prompt = ("pos", (1966, 620, 2118, 816))
    
    # 4. 构造输入对象
    # 此时 text_prompt 传空字符串 ""
    input_obj = trtsam3.Sam3Input(image, "", [box_prompt])

    # 5. 推理
    start = time.time()
    results = engine.forward(input_obj)
    end = time.time()
    print(f"Inference time: {(end-start)*1000:.2f} ms")

    # 6. 可视化
    # 先画 Prompt Box (蓝色)
    x1, y1, x2, y2 = map(int, box_prompt[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    for obj in results:
        mask = obj.segmentation.mask
        if mask is not None:
            colored_mask = np.zeros_like(image)
            colored_mask[:, :, 1] = 255 # Green
            image[mask > 0] = cv2.addWeighted(image[mask > 0], 0.5, colored_mask[mask > 0], 0.5, 0)
        
        # 预测框
        b = obj.box
        cv2.rectangle(image, (int(b.left), int(b.top)), (int(b.right), int(b.bottom)), (0, 255, 0), 2)

    cv2.imwrite("output/py_box_result.jpg", image)
    print("Saved result to output/py_box_result.jpg")

def run_visualization():
    engine, tokenizer = get_engine_and_tokenizer(gpu_id=1)

    prompt_text = "person"
    encoded = tokenizer.encode(prompt_text)
    engine.setup_text_inputs(prompt_text, encoded.ids, encoded.attention_mask)

    image1 = cv2.imread("images/persons.jpg")
    if image1 is None:
        print("Image not found")
        return

    image2 = cv2.imread("images/smx.jpg")
    if image2 is None:
        print("Image not found")
        return
    
    # 单图推理
    input_obj1 = trtsam3.Sam3Input(image1, prompt_text)
    input_obj2 = trtsam3.Sam3Input(image2, prompt_text)
    results = engine.forwards([input_obj1, input_obj2])
    
    # 可视化
    osd_image = osd(image1.copy(), results[0])
    cv2.imwrite("output/py_result1.jpg", osd_image)
    print("Saved result to output/py_result.jpg")

    # 可视化
    osd_image = osd(image2.copy(), results[1])
    cv2.imwrite("output/py_result2.jpg", osd_image)
    print("Saved result to output/py_result.jpg")

if __name__ == "__main__":
    run_box_prompt_test()
    run_visualization()
    speed_test()