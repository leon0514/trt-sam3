from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BoxInput(BaseModel):
    label: str = Field(default="pos", description="'pos' for positive, 'neg' for negative")
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates", min_length=4, max_length=4)

# 1个text + 多个boxes 的组合
class PromptUnit(BaseModel):
    text: Optional[str] = Field(default="", description="Text prompt")
    boxes: Optional[List[BoxInput]] = Field(default=[], description="Geometric boxes")

# API 输入输出模型
# 图片数据采用 Base64 编码字符串
# 阈值
# 多个 Prompt 单元
# 是否返回掩码
class CropConfig(BaseModel):
    max_size: int = Field(default=640, description="Max crop size")
    padding: int = Field(default=20, description="Crop padding")
    w_diou: float = Field(default=30.0, description="Distance penalty weight")
    w_expansion: float = Field(default=5.0, description="Expansion penalty weight")
    count_penalty: float = Field(default=120.0, description="Crop count penalty")
    nms_threshold: float = Field(default=0.2, description="Overlap NMS threshold")
    enable_ar_fix: bool = Field(default=True, description="Enable aspect ratio fix")
    target_ar: float = Field(default=1.0, description="Target aspect ratio")

class InferenceRequest(BaseModel):
    image_base64: str = Field(default="", description="Base64 encoded image data")
    confidence_threshold: float = Field(default=0.5, description="object confidence threshold")
    pre_detect_labels: List[str] = Field(default=["person"], description="Pre-detect labels for cropping")
    prompts: List[PromptUnit] = Field(..., description="List of prompts")
    return_mask: bool = Field(default=False, description="If True, returns segmentation masks")
    merge_results: bool = Field(default=True, description="If True, merge full image and crop results")
    crop_config: Optional[CropConfig] = Field(default=None, description="Omnicrop configuration")

class DetectionResult(BaseModel):
    label: str
    score: float
    box: List[float]
    mask: Optional[Dict] = None

class InferenceResponse(BaseModel):
    results: List[DetectionResult]