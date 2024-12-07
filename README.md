# NORI_image_segmentation

docker run -v ${PWD}\work_directory\train_directory:/train_models/train_directory -v ${PWD}\work_directory\datasets:/train_models/datasets  -d nori_image_segmentation:latest tubule_model.py

docker run --gpus all -v ${PWD}\work_directory\train_directory:/train_models/train_directory -v ${PWD}\work_directory\datasets:/train_models/datasets  -d nori_image_segmentation:latest tubule_model.py