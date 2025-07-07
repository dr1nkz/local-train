from ultralytics import YOLO

Yolo = YOLO('PATH_TO_MODEL')
Yolo.predict('PATH_TO_IMAGE', save=True, imgsz=640, conf=0.5)
