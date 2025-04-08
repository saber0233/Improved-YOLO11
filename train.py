from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO(r'yolo11-FSPPF-DySample-SCC3k2.yaml')
    # model = YOLO(r'yolo11n.yaml').load("weights/yolo11n.pt")  # Train with initial weights

    # Training parameters ----------------------------------------------------------------------------------------------
    model.train(
        data=r'',  # (str) Path to dataset config file
        epochs=100,  # (int) Number of epochs to train for
        patience=50,  # (int) Number of epochs to wait for no improvement before early stopping
        batch=32,  # (int) Number of images per batch (-1 for auto batch)
        imgsz=640,  # (int) Image size for training; either a single integer or (w, h)
        save=True,  # (bool) Save training checkpoints and predictions
        save_period=-1,  # (int) Save checkpoints every x epochs (disable if <1)
        cache=False,  # (bool) Cache images in RAM/disk or disable caching
        device='',  # (int | str | list, optional) Device to run on, e.g. 'cuda:0', '0,1,2,3', or 'cpu'
        workers=8,  # (int) Number of data loading workers per DDP process
        project='runs/train',  # (str, optional) Project name
        name='exp',  # (str, optional) Experiment name; results saved to 'project/name'
        exist_ok=False,  # (bool) Overwrite existing experiment if True
        pretrained=True,  # (bool | str) Use pretrained model or load weights from given model path
        optimizer='AdamW',  # (str) Optimizer to use, options: [SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
        verbose=True,  # (bool) Enable detailed logging
        seed=0,  # (int) Random seed for reproducibility
        deterministic=True,  # (bool) Enable deterministic training
        single_cls=False,  # (bool) Train multi-class data as single-class
        rect=False,  # (bool) Use rectangular training for training or rectangular validation for validation mode
        cos_lr=False,  # (bool) Use cosine learning rate scheduler
        close_mosaic=0,  # (int) Disable mosaic augmentation for last N epochs
        resume=False,  # (bool) Resume training from last checkpoint
        amp=True,  # (bool) Enable Automatic Mixed Precision (AMP) training
        fraction=1.0,  # (float) Fraction of training dataset to use (default is 1.0 for all images)
        profile=False,  # (bool) Enable ONNX and TensorRT profiling during training
        freeze=None,  # (int | list, optional) Freeze first n layers or a list of layer indices during training

        # Segmentation
        overlap_mask=True,  # (bool) Whether to allow overlapping masks during training (segmentation only)
        mask_ratio=4,  # (int) Downsampling ratio for masks (segmentation only)

        # Classification
        dropout=0.0,  # (float) Dropout regularization rate (classification only)

        # Hyperparameters ----------------------------------------------------------------------------------------------
        lr0=0.001,  # (float) Initial learning rate (e.g., SGD=1E-2, Adam=1E-3)
        lrf=0.0001,  # (float) Final learning rate (lr0 * lrf)
        momentum=0.9,  # (float) SGD momentum / Adam beta1
        weight_decay=0.0005,  # (float) Weight decay for optimizer
        warmup_epochs=3.0,  # (float) Warmup epochs (can be fractional)
        warmup_momentum=0.8,  # (float) Initial momentum during warmup
        warmup_bias_lr=0.1,  # (float) Initial bias learning rate during warmup
        box=7.5,  # (float) Box loss gain
        cls=0.5,  # (float) Class loss gain (scaled by pixels)
        dfl=1.5,  # (float) Distribution Focal Loss gain
        pose=12.0,  # (float) Pose loss gain
        kobj=1.0,  # (float) Keypoint object loss gain
        label_smoothing=0.0,  # (float) Label smoothing (fraction)
        nbs=64,  # (int) Nominal batch size

        # Data augmentation --------------------------------------------------------------------------------------------
        hsv_h=0.0,  # (float) HSV-Hue augmentation (fraction)
        hsv_s=0.0,  # (float) HSV-Saturation augmentation (fraction)
        hsv_v=0.0,  # (float) HSV-Value augmentation (fraction)
        degrees=0.0,  # (float) Image rotation (+/- degrees)
        translate=0.0,  # (float) Image translation (+/- fraction)
        scale=0.0,  # (float) Image scaling (+/- gain)
        shear=0.0,  # (float) Image shearing (+/- degrees)
        perspective=0.0,  # (float) Perspective transform (+/- fraction), range 0-0.001
        flipud=0.0,  # (float) Vertical flip probability
        fliplr=0.0,  # (float) Horizontal flip probability
        mosaic=0.0,  # (float) Mosaic augmentation probability
        mixup=0.0,  # (float) MixUp augmentation probability
        copy_paste=0.0,  # (float) Copy-Paste augmentation for segmentation (probability)
    )
