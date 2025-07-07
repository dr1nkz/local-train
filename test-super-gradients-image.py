from super_gradients.training import models
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
yolo_nas_model = models.get('yolo_nas_m', num_classes=1,
  checkpoint_path="path to /checkpoints/test/average_model.pth"
  ).to(device)
yolo_nas_model.predict('PATH_TO_IMAGE').show()