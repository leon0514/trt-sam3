import requests
import base64
import cv2
import numpy as np
import json
import random
import os

# 服务地址
SERVER_URL = "http://172.16.20.193:18002/predict"
OUTPUT_DIR = "client_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rle_to_opencv_mask(rle_dict):
    """
    核心函数：将 RLE 字典转换为 OpenCV (numpy) 二值掩码
    
    Args:
        rle_dict: {'size': [h, w], 'counts': [c1, c2, ...]}
    Returns:
        mask: numpy array of shape (h, w), dtype=uint8 (0 or 255)
    """
    h, w = rle_dict['size']
    counts = rle_dict['counts']
    
    # 1. 创建扁平数组
    total_pixels = h * w
    mask_flat = np.zeros(total_pixels, dtype=np.uint8)
    
    # 2. 填充像素
    # 假设 counts 格式为 [背景数, 前景数, 背景数, 前景数...]
    # 如果第一个像素是前景，服务端通常会在 counts 开头补 0
    current_pos = 0
    val = 0 # 0 表示背景，1 表示前景
    
    for count in counts:
        if val == 1:
            # 只有当前是前景时才赋值
            mask_flat[current_pos : current_pos + count] = 1
        current_pos += count
        val = 1 - val # 切换状态
        
    # 3. 重塑回图像尺寸
    # 注意：服务端使用的是 flatten(order='F') (列优先)，所以这里也要用 order='F'
    mask = mask_flat.reshape((h, w), order='F')
    
    return mask * 255 # 转换为 0-255 方便 OpenCV 显示

def get_random_color(tag):
    random.seed(hash(tag))
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def visualize(image, results, save_name="result.jpg"):
    """
    可视化结果 (适配 Box-based Mask)
    """
    if image is None: return
    vis_img = image.copy()
    print(f"Detected {len(results)} objects.")

    for obj in results:
        label = obj['label']
        score = obj['score']
        box = obj['box'] # [x1, y1, x2, y2]
        rle = obj.get('mask')
        
        # 生成随机颜色
        color = get_random_color(label)
        
        # 1. 绘制 Box
        x1, y1, x2, y2 = map(int, box)
        
        # 边界检查，防止坐标超出图像范围
        h_img, w_img = vis_img.shape[:2]
        x1 = max(0, min(x1, w_img))
        y1 = max(0, min(y1, h_img))
        x2 = max(0, min(x2, w_img))
        y2 = max(0, min(y2, h_img))
        
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        
        # 2. 绘制 Mask (关键修改点)
        if rle:
            # 解码 Mask (此时得到的 mask_cv 是 Box 大小的，比如 91x50)
            mask_cv = rle_to_opencv_mask(rle)
            
            # 计算 Box 的宽高
            box_w = x2 - x1
            box_h = y2 - y1
            
            # 只有当 Box 有效时才绘制
            if box_w > 0 and box_h > 0:
                # 尺寸容错：由于 float->int 转换误差，C++返回的mask尺寸可能和Python切片尺寸差1-2像素
                # 强制 resize mask 到 Python 切片的大小
                if mask_cv.shape[0] != box_h or mask_cv.shape[1] != box_w:
                    mask_cv = cv2.resize(mask_cv, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                
                # 提取原图的 ROI (Region of Interest)
                roi = vis_img[y1:y2, x1:x2]
                
                # 创建彩色遮罩层
                colored_mask = np.zeros_like(roi)
                colored_mask[:] = color
                
                # 找到前景区域
                mask_indices = mask_cv > 0
                
                # 仅在 ROI 内进行半透明叠加
                # 错误修正：这里 mask_indices 和 roi 的维度现在完全匹配了
                alpha = 0.5
                roi[mask_indices] = cv2.addWeighted(
                    roi[mask_indices], 1 - alpha, 
                    colored_mask[mask_indices], alpha, 0
                )
                
                # 将处理后的 ROI 放回原图
                vis_img[y1:y2, x1:x2] = roi

        # 3. 绘制标签
        text = f"{label}: {score:.2f}"
        cv2.putText(vis_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    save_path = os.path.join(OUTPUT_DIR, save_name)
    cv2.imwrite(save_path, vis_img)
    print(f"Result saved to {save_path}")

def run_inference(image_path, prompts, return_mask=True):
    # 1. 读取图片并转 Base64
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    # 2. 构造请求 Payload
    payload = {
        "image_base64": img_b64,
        "prompts": prompts,
        "return_mask": return_mask
    }

    # 3. 发送请求
    try:
        print(f"Sending request for {image_path} ...")
        resp = requests.post(SERVER_URL, json=payload)
        
        if resp.status_code == 200:
            data = resp.json()
            results = data['results']
            img = cv2.imread(image_path)
            visualize(img, results, save_name="client_" + os.path.basename(image_path))
        else:
            print(f"Request failed: {resp.status_code}")
            print(resp.text)

    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    # 场景 A: 多文本提示 (Multi-Class)
    # 同时检测人、眼镜、汽车
    img_path = "../images/persons.jpg" # 请确保图片存在
    prompts_a = [
        {"text": "hat"},
    ]
    run_inference(img_path, prompts_a, return_mask=True)

    # 场景 B: 混合提示 (文本 + 框)
    # 检测 mask，同时强制分割某个区域
    img_path_b = "../images/smx.jpg" # 请确保图片存在
    prompts_b = [
        {
            "text": "helmet"
        },
        {
            "text": "", # 空文本，只用框
            "boxes": [
                # 这里的坐标需要根据你的图片实际情况修改
                {"label": "pos", "bbox": [100, 100, 400, 500]} 
            ]
        }
    ]
    run_inference(img_path_b, prompts_b, return_mask=True)
