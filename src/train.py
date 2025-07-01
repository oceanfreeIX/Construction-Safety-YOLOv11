from ultralytics import YOLO
import torch
# 添加多进程保护块
if __name__ == '__main__':
    # GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device} | GPU名称: {torch.cuda.get_device_name(0)}")
    model = YOLO("../runs/yolo11n.pt").to(device)
    # 训练参数
    results = model.train(
        data="construction.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        patience=30,
        device=0,
        name="yolov11_construction",
        seed=42,
        workers=0  # 关键修改：禁用多进程加载器
    )