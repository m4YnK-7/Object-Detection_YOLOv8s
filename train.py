from ultralytics import YOLO
import os

config = {
            # Training parameters
            'epochs': 10,  # Increased for better convergence
            'batch_size': 16,  # Adjust based on GPU memory
            'imgsz': 256,   # Increased from 128 for better detection
            'device': 'cpu',
            
            # Optimizer settings
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,    # Higher final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            
            # Augmentation (critical for challenging conditions)
            'hsv_h': 0.015,     # Hue augmentation for lighting variations
            'hsv_s': 0.7,       # Saturation for lighting conditions
            'hsv_v': 0.4,       # Value/brightness for lighting
            'degrees': 15.0,    # Rotation for different camera angles
            'translate': 0.2,   # Translation augmentation
            'scale': 0.9,       # Scale variation for different distances
            'shear': 2.0,       # Shear transformation
            'perspective': 0.0003,  # Perspective transformation
            'flipud': 0.5,      # Vertical flip (useful in zero-gravity)
            'fliplr': 0.5,      # Horizontal flip
            'mosaic': 0.8,      # Increased mosaic for occlusion handling
            'mixup': 0.10,      # Mixup augmentation for robustness
            'copy_paste': 0.3,  # Copy-paste augmentation for occlusions
            
            # NMS settings
            'iou': 0.7,         # IoU threshold for NMS
            'conf': 0.25,       # Confidence threshold
            
            # Model architecture
            'dropout': 0.1,     # Dropout for regularization
}

if __name__ == '__main__': 
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov8m.pt"))

    results = model.train(
            data=os.path.join(this_dir, "yolo_params.yaml"),
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            device=config['device'],
            
            # Optimizer settings
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
                     
                
            # Validation settings
            iou=config['iou'],
            conf=config['conf'],
            
            # Additional settings
            dropout=config['dropout'],
            save_period=8,  # Save checkpoint every 10 epochs
            cache="disk",      # Cache images for faster training
            workers=8,       # Number of dataloader workers
            exist_ok=True,
            pretrained=True,
            verbose=True,
            close_mosaic=4
        )

    
    # tuning = model.tune(
    #     data="yolo_params.yaml",           # Path to your dataset config
    #     epochs=5 ,                  # Training epochs per trial
    #     imgsz=128,                  # Image size
    #     project="yolo_tune_project",# Optional: custom project folder
    #     name="exp_tune",                       # Optional: experiment name
    #     iterations=100,
    # )
    
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''