from ultralytics import YOLO

model = YOLO("yolov8n.yaml")


if __name__ == '__main__':
    results = model.train(data="YOLO_model\data.yaml", epochs=500, patience=0)