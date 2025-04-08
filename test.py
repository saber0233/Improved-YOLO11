from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'yolo11n.yaml')  # build a new model from YAML
    model.info()

