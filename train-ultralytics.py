from ultralytics import YOLO

Yolo = YOLO('yolov9c.pt')
results = Yolo.train(data='./pigs.yaml', epochs=50, imgsz=640, workers=4)

Yolo = YOLO('/kaggle/working/runs/detect/train/weights/best.pt')
Yolo.val(data='./pigs.yaml', split='test', imgsz=640)
Yolo.export(format='onnx', opset=12)
