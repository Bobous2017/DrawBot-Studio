FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git ffmpeg potrace libcairo2 \
    libgl1 libgl1-mesa-dri mesa-utils \
    libosmesa6 xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/facebookresearch/AnimatedDrawings.git /tmp/animated_drawings \
    && cd /tmp/animated_drawings \
    && pip install -e . \
    && cd /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]