from ultralytics import YOLO
import os
import cv2  # 添加OpenCV用于保存图像

def simple_test():
    """简化版测试脚本"""
    # 加载模型
    model_path = "../models/best.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 设置测试源
    test_source = "construction_dataset/test/images"
    if not os.path.exists(test_source):
        print(f"⚠️ 警告：测试集路径不存在 {test_source}")
        print("将使用验证集进行测试演示")
        test_source = "construction_dataset/val/images"

    # 执行预测
    print("开始执行预测...")
    results = model.predict(
        source=test_source,
        conf=0.3,
        save=True,  # Ultralytics会自动保存结果
        save_txt=True,
        workers=0  # 禁用多进程
    )

    # 手动收集结果并保存示例图像
    print("\n测试完成！以下是预测统计：")
    total_objects = 0
    detected_classes = {}

    # 创建目录保存自定义结果
    os.makedirs("../test_results", exist_ok=True)

    # 处理前3个结果
    for i, r in enumerate(results[:3]):
        # 获取带标注的图像（NumPy数组）
        annotated_img = r.plot()  # 这是NumPy数组

        # 使用OpenCV保存图像
        output_path = f"test_results/result_{i + 1}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        print(f"保存结果图片: {output_path}")

        # 统计信息
        total_objects += len(r.boxes)
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            detected_classes[cls_name] = detected_classes.get(cls_name, 0) + 1

    # 打印统计信息
    print(f"- 测试图片数量: {len(results)}")
    print(f"- 检测到目标总数: {total_objects}")
    print("- 各类别检测数量:")
    for cls_name, count in detected_classes.items():
        print(f"  {cls_name}: {count}")

    # 创建README文件
    with open("../test_results/README.txt", "w") as f:
        f.write("施工安全检测模型测试结果\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试数据集: {test_source}\n")
        f.write(f"测试图片数量: {len(results)}\n")
        f.write(f"检测到目标总数: {total_objects}\n\n")
        f.write("各类别检测数量:\n")
        for cls_name, count in detected_classes.items():
            f.write(f"  {cls_name}: {count}\n")
        f.write("\n结果图片保存在当前目录")


if __name__ == '__main__':
    print("=" * 50)
    print("施工安全检测模型测试")
    print("=" * 50)

    simple_test()
    print("\n✅ 测试完成！结果保存在 test_results 目录")
    print("=" * 50)