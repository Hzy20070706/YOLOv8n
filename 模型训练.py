from ultralytics import YOLO

def main():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练模型
    results = model.train(
        data='data.yaml',  # 数据集配置文件
        epochs=200,  # 训练轮次
        imgsz=640,  # 图片尺寸
        batch=4,  # 批次大小（适合小数据集）
        patience=30,  # 早停耐心值，防止过拟合
        cache=True,  # 缓存图片到内存，加速训练
        device=0,  # 使用GPU训练（-1表示CPU）
        workers=0,  # 数据加载线程数（0表示单线程，Windows建议0）

        # 增强小数据集泛化能力
        mosaic=1.0,  # 启用马赛克增强（0.0-1.0）
        mixup=0.1,  # 启用mixup增强（0.0-1.0）
        copy_paste=0.1,  # 启用复制粘贴增强（0.0-1.0）

        # 正则化防止过拟合
        weight_decay=0.0005,  # 权重衰减
        dropout=0.1,  # Dropout概率（仅YOLOv8n支持）

        # 学习率策略
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率因子
        momentum=0.937,  # SGD动量

        # 输出设置
        project='runs/train',  # 训练结果保存路径
        name='exp',  # 实验名称
        exist_ok=True,  # 如果存在则覆盖
        verbose=True  # 显示详细日志
    )

    # 训练完成后自动评估
    metrics = model.val()
    print("训练完成，评估结果：", metrics)

# 关键：将主逻辑放入这个代码块中，避免Windows多进程报错
if __name__ == '__main__':
    main()