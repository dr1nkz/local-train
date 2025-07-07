from super_gradients.training import models
import torch
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
yolo_nas_model = models.get('yolo_nas_m', num_classes=2,
                            checkpoint_path="average_model.pth"
                            ).to(device)
out = yolo_nas_model.predict('1_resized.mp4', fuse_model=False)
out.save('predicted.mp4')
