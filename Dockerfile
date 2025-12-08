FROM nvcr.io/nvidia/tensorrt:25.10-py3

RUN apt update -y && apt install libopencv-dev libfreetype6-dev -y

RUN pip install opencv-python-headless fastapi numpy tokenizers uvicorn aiohttp -i https://pypi.tuna.tsinghua.edu.cn/simpleleon@1234