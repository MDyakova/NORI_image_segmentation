# Load base image
FROM python:3.8

# Install libraries
COPY settings /settings
RUN chmod -R 777 /settings

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /settings/requirements.txt