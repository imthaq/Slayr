FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgl1 libgles2 libegl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
