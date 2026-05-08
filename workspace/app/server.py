from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

# 导入 loguru
from loguru import logger

import inference
import router_ui
import router_predict
import router_refine

# --- Loguru 日志配置 ---

# 1. 移除 FastAPI 默认的日志处理器，以便 loguru 完全接管
logger.remove()

# 2. 添加控制台输出
#    - level="INFO": 设置日志级别
#    - colorize=True: 开启颜色显示
#    - format: 自定义日志格式，包含了时间、级别、模块、函数、行号和消息
logger.add(
    sys.stderr, 
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 3. 添加文件输出
#    - "logs/app_{time:YYYY-MM-DD}.log": 日志文件名模板，每天生成一个新文件
#    - rotation="10 MB": 当文件大小超过 10MB 时，自动切分
#    - retention="7 days": 最多保留 7 天的日志文件
#    - compression="zip": 自动将旧的日志文件压缩为 .zip 格式
#    - encoding="utf-8": 使用 UTF-8 编码
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, "app_{time:YYYY-MM-DD}.log"), 
    level="INFO",
    rotation="10 MB", 
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

logger.info("Logger initialized successfully.")


# 定义 lifespan 来替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup 逻辑 ---
    logger.info("Application startup...")

    # 确保文件夹存在
    logger.info(f"Ensuring directory exists: {inference.UPLOADS_DIR}")
    os.makedirs(inference.UPLOADS_DIR, exist_ok=True)
    logger.info(f"Ensuring directory exists: {inference.OUTPUT_DIR}")
    os.makedirs(inference.OUTPUT_DIR, exist_ok=True)
    
    # 模型单例加载
    logger.info("Initializing ModelManager...")
    try:
        inference.ModelManager.initialize(gpu_id=0)
        logger.info("ModelManager initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ModelManager: {e}")
        # 在关键组件加载失败时，可以选择退出应用
        # raise SystemExit("Critical component failed to load. Exiting.")
    
    yield  # 这里是应用运行的时间点
    
    # --- Shutdown 逻辑 ---
    logger.info("Application shutting down...")

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
    logger.info("Root endpoint '/' was hit, serving index.html.")
    return FileResponse('frontend/index.html')

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)