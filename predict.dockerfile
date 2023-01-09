# Base image
FROM  nvcr.io/nvidia/pytorch:22.12-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts of the application from repository
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Set working directory and install requirements
WORKDIR /
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Define entrypoint (what is run when the image is executed)
ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]
