FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY endoai/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: install Jupyter for notebooks
RUN pip install jupyterlab

# Copy the rest of the project
COPY . .

# Default command
CMD ["bash"]
