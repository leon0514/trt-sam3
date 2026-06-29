import cv2
import numpy as np
import base64
from fastapi import HTTPException


def decode_b64(b64_str):
    try:
        data = base64.b64decode(b64_str.split(',')[-1])
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except:
        raise HTTPException(status_code=400, detail="Base64 decode failed")


def binary_mask_to_frontend_rle(mask: np.ndarray) -> list:
    """
    将二值 mask 转换为前端 canvas 可直接解码的扁平 RLE。

    输出格式：[start1, len1, start2, len2, ...]
      - start 为 1-based 像素索引
      - 像素按 HTML canvas/ImageData 行主序（row-major）排列，
        即 idx = y * width + x
    """
    if mask is None:
        return []

    # 行主序，与前端 canvas createImageData 一致
    flat = mask.ravel(order='C')
    flat = (flat > 0).astype(np.int8)

    ones = np.where(flat == 1)[0]
    if ones.size == 0:
        return []

    diffs = np.diff(ones)
    break_indices = np.where(diffs > 1)[0]

    rle = []
    start = int(ones[0])
    for idx in break_indices:
        end = int(ones[idx])
        rle.extend([start + 1, end - start + 1])
        start = int(ones[idx + 1])
    rle.extend([start + 1, int(ones[-1]) - start + 1])

    return rle
