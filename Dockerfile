FROM nvcr.io/nvidia/tritonserver:25.04-py3

# Set working directory
WORKDIR /app/models

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Expose Triton ports (helps when running without -p manually)
EXPOSE 8000 8001 8002

# Set default command (overridden when using `docker run`)
CMD ["tritonserver", "--model-repository=/app/models/model_repository"]
