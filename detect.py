from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO(r'weights/yolo11n.pt')  # Path to weights file

    # Run prediction
    model.predict(
        source=r'ultralytics/assets/bus.jpg',  # Path to image/video/file/folder/URL/stream
        save=True,  # Save prediction results
        imgsz=640,  # Input image size, can be an int or (w, h)
        conf=0.25,  # Confidence threshold for object detection (default is 0.25 for prediction, 0.001 for validation)
        iou=0.45,  # IoU threshold for Non-Maximum Suppression (NMS)
        show=False,  # Show results if possible
        project='runs/predict',  # Project name (optional)
        name='exp',  # Experiment name; results are saved in 'project/name' (optional)
        save_txt=False,  # Save results as .txt files
        save_conf=True,  # Save results with confidence scores
        save_crop=False,  # Save cropped images and results
        show_labels=True,  # Display object labels on image
        show_conf=True,  # Display object confidence scores on image
        vid_stride=1,  # Video frame stride
        line_width=3,  # Line thickness of bounding boxes (pixels)
        visualize=False,  # Visualize model features
        augment=False,  # Apply image augmentation to prediction source
        agnostic_nms=False,  # Class-agnostic NMS
        retina_masks=False,  # Use high-resolution segmentation masks
        boxes=True,  # Show bounding boxes in segmentation predictions
    )
