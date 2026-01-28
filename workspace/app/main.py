from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os

import inference
import router_ui
import router_predict
import router_refine

# 定义 lifespan 来替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup 逻辑 ---
    # 确保文件夹存在
    os.makedirs(inference.UPLOADS_DIR, exist_ok=True)
    os.makedirs(inference.OUTPUT_DIR, exist_ok=True)
    # 模型单例加载
    inference.ModelManager.initialize(gpu_id=1)
    
    yield  # 这里是应用运行的时间点
    
    # --- Shutdown 逻辑 (如果需要可以写在这里) ---
    print("Shutting down...")

app = FastAPI(title="TRTSAM3 System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(router_ui.router)
app.include_router(router_predict.router)
app.include_router(router_refine.router)

# 静态文件处理
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)