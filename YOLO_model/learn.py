from ultralytics import YOLO

model = YOLO("yolov8s.yaml").load("yolov8s.pt")


if __name__ == '__main__':
    results = model.train(data="YOLO_model\data.yaml", epochs=25, plots=True, optimizer = 'SGD', dropout =0.5 )
