import random
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import zipfile
import requests
import os
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import dataloaders
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.transforms.transforms import (
    DetectionMosaic,
    DetectionHSV,
    DetectionRandomAffine
)
import albumentations as A
import torch

# Save the original torch.load function
_original_torch_load = torch.load

# Define a new function that forces weights_only=False


def custom_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


# Override torch.load globally
torch.load = custom_torch_load


dataset_params = {
    'data_dir': '/home/pc/Downloads/local-train/dataset_pigs',
    'train_images_dir': '/home/pc/Downloads/local-train/dataset_pigs/train/images',
    'train_labels_dir': '/home/pc/Downloads/local-train/dataset_pigs/train/labels',
    'val_images_dir': '/home/pc/Downloads/local-train/dataset_pigs/valid/images',
    'val_labels_dir': '/home/pc/Downloads/local-train/dataset_pigs/valid/labels',
    'test_images_dir': '/home/pc/Downloads/local-train/dataset_pigs/valid/images',
    'test_labels_dir': '/home/pc/Downloads/local-train/dataset_pigs/valid/labels',
    'classes': ['pig', 'human']  # изменено на 2 класса
}

# Трансформации для обучения
train_transforms = [
    DetectionMosaic(prob=1.0, input_dim=(640, 640)),
    DetectionHSV(prob=0.5),
    DetectionRandomAffine(degrees=10, translate=0.1, shear=2)
]

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes'],
        'transforms': train_transforms
    },
    dataloader_params={
        'batch_size': 16,
        'num_workers': 8
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
        # обычно валидацию без агрессивных аугментаций
    },
    dataloader_params={
        'batch_size': 16,
        'num_workers': 8
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': 16,
        'num_workers': 8
    }
)

train_params = {
    'silent_mode': False,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 3e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "SGD",
    "optimizer_params": {"momentum": 0.937, "weight_decay": 5e-4},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"beta": 0.9998, "decay_type": "exp"},
    "max_epochs": 150,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

model = models.get('yolo_nas_m', num_classes=len(dataset_params['classes']))
trainer = Trainer(experiment_name='test', ckpt_root_dir='checkpoints')
trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data
)

trainer.test(
    model=model,
    test_loader=test_data,
    test_metrics_list=train_params["valid_metrics_list"]
)
