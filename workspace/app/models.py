from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BoxInput(BaseModel):
    label: str = Field(default="pos", description="'pos' for positive, 'neg' for negative")
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates", min_length=4, max_length=4)

class PromptUnit(BaseModel):
    text: Optional[str] = Field(default="", description="Text prompt")
    boxes: Optional[List[BoxInput]] = Field(default=[], description="Geometric boxes")

class InferenceRequest(BaseModel):
    image_base64: str = Field(default="", description="Base64 encoded image data")
    confidence_threshold: float = Field(default=0.5, description="object confidence threshold")
    prompts: List[PromptUnit] = Field(..., description="List of prompts")
    return_mask: bool = Field(default=False, description="If True, returns segmentation masks")

class DetectionResult(BaseModel):
    label: str
    score: float
    box: List[float]
    mask: Optional[Dict] = None

class InferenceResponse(BaseModel):
    results: List[DetectionResult]