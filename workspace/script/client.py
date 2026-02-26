import requests
import base64
import json
import time
import os
import cv2
import numpy as np

class TRTSAMClient:
    def __init__(self, host="172.16.20.193", port=18001):
        self.base_url = f"http://{host}:{port}"
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

    def _img_to_b64(self, image_path):
        """将本地图片转为 base64 字符串"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"未找到图片文件: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def rle_to_opencv_mask(self, rle_dict):
        """还原 RLE 格式的 Mask (Fortran order)"""
        h, w = rle_dict['size']
        counts = rle_dict['counts']
        total_pixels = h * w
        mask_flat = np.zeros(total_pixels, dtype=np.uint8)
        
        current_pos = 0
        val = 0 
        for count in counts:
            if val == 1:
                mask_flat[current_pos : current_pos + count] = 1
            current_pos += count
            val = 1 - val 
            
        mask = mask_flat.reshape((h, w), order='F')
        return mask * 255

    def infer(self, endpoint, image_path, prompts, conf=0.5, mask=False):
        """
        发送推理请求
        :param endpoint: API 后缀，如 '/predict'
        :param prompts: List[dict], 例如 [{"text": "person", "boxes": []}, {"text": "bag", "boxes": []}]
        :param conf: 置信度阈值
        :param mask: 是否返回掩码
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        payload = {
            "image_base64": self._img_to_b64(image_path),
            "confidence_threshold": conf,
            "prompts": prompts, # 直接透传 prompts 列表
            "return_mask": mask
        }

        try:
            start_time = time.time()
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                res = response.json()
                # 打印原始结果以便调试
                print(f"[{endpoint}] 耗时: {elapsed:.2f}s | 结果数: {len(res.get('results', []))}")
                return res
            else:
                print(f"请求失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"网络异常: {e}")
            return None

    def visualize(self, image_path, result, save_name="vis_result.jpg"):
        """可视化检测框和局部 Mask"""
        objects = result.get('results', [])
        img = cv2.imread(image_path)
        if img is None: return
        
        h_img, w_img = img.shape[:2]
        mask_overlay = img.copy()
        
        for obj in objects:
            # 1. 绘制矩形框
            box = obj.get('box', [])
            if len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                # 限制坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_str = f"{obj.get('label', 'obj')} {obj.get('score', 0):.2f}"
                cv2.putText(img, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 2. 处理 Mask (局部 ROI 贴图)
                rle = obj.get('mask')
                if rle:
                    binary_mask = self.rle_to_opencv_mask(rle)
                    m_h, m_w = binary_mask.shape
                    
                    # 后端返回的局部 Mask 对应的实际区域
                    t_y2 = min(y1 + m_h, h_img)
                    t_x2 = min(x1 + m_w, w_img)
                    
                    # 确保切片大小一致
                    roi = mask_overlay[y1:t_y2, x1:t_x2]
                    cut_m = binary_mask[0:(t_y2-y1), 0:(t_x2-x1)]
                    
                    color = [np.random.randint(100, 255) for _ in range(3)]
                    roi[cut_m > 0] = color
        
        final = cv2.addWeighted(mask_overlay, 0.5, img, 0.5, 0)
        cv2.imwrite(save_name, final)
        print(f"结果已保存至: {save_name}")

# ==========================================
# 运行示例
# ==========================================
if __name__ == "__main__":
    client = TRTSAMClient()
    
    # 定义多个 Prompt
    my_prompts = [
        {"text": "person", "boxes": []},
        {"text": "cigarette", "boxes": []},
    ]
    
    img_path = "../images/smoke.jpg"
    
    res = client.infer(
        endpoint="/predict-obj-about-small-object", 
        image_path=img_path,
        prompts=my_prompts, # 传入列表
        conf=0.4,
        mask=True
    )
    
    if res:
        client.visualize(img_path, res, "output_multi_prompt.jpg")