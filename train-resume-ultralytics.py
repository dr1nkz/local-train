from ultralytics import YOLO

model = YOLO("PATH_TO_LAST.pt")  # load a partially trained model

results = model.train(resume=True)
