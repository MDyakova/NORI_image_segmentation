# Load base image
FROM python:3.8

# Install libraries
COPY settings /settings
RUN chmod -R 777 /settings

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /settings/requirements.txt && \
    apt-get update && apt-get install -y libgl1-mesa-glx

# Copy inference directory
COPY inference_models /inference_models
RUN chmod -R 777 /inference_models
WORKDIR /inference_models

# Run unit-tests
RUN python3 -m pytest unit_tests.py

# Launch code
ENTRYPOINT ["python"]
CMD ["main.py"]