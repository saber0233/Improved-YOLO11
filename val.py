from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO(r'weights/yolo11n.pt')   # Path of weight.py

    # Validate the model
    metrics = model.val(
        val=True,  # (bool) Perform validation/testing during training
        data=r'',  # (str) Path to dataset configuration file
        split='test',  # (str) Dataset split to use for validation, e.g., 'val', 'test', or 'train'
        batch=1,  # (int) Number of images per batch (-1 for auto batch size)
        imgsz=640,  # (int) Input image size, can be a single int or (w, h)
        device='',  # (str) Device to run on, e.g., 'cuda:0', '0,1,2,3', or 'cpu'
        workers=1,  # (int) Number of data loading workers per DDP process
        save_json=False,  # (bool) Save results to a JSON file
        save_hybrid=False,  # (bool) Save hybrid version of labels (labels + extra predictions)
        conf=0.001,  # (float) Confidence threshold for detection (default 0.25 for prediction, 0.001 for validation)
        iou=0.6,  # (float) IoU threshold for Non-Maximum Suppression (NMS)
        project='runs/test',  # (str, optional) Project name
        name='exp',  # (str, optional) Experiment name; results saved to 'project/name'
        max_det=30,  # (int) Maximum number of detections per image
        half=False,  # (bool) Use half precision (FP16)
        dnn=False,  # (bool) Use OpenCV DNN for ONNX inference
        plots=True,  # (bool) Save plots during training/validation
    )

    # Print evaluation metrics
    print(f"mAP50-95: {metrics.box.map}")  # mAP@[.50:.95]
    print(f"mAP50: {metrics.box.map50}")  # mAP@0.50
    print(f"mAP75: {metrics.box.map75}")  # mAP@0.75

    # Calculate FPS
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}")  # Frames Per Second
