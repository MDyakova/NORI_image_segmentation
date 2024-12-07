# Load base image
FROM python:3.8

# Install libraries
COPY settings /settings
RUN chmod -R 777 /settings

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /settings/requirements.txt && \
    apt-get update && apt-get install -y libgl1-mesa-glx


# Copy train directory
COPY train_models /train_models
RUN chmod -R 777 /train_models
WORKDIR /train_models

# Launch code
ENTRYPOINT ["python"]
CMD ["tubule_model.py"]