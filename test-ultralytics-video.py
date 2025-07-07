from ultralytics import YOLO

Yolo = YOLO('PATH_TO_MODEL')
Yolo.predict('PATH_TO_VIDEO', save=True)
