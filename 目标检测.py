import cv2
import numpy as np
import mss
import win32gui
import win32con
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import time
import yaml  # 用于读取YAML配置文件

# ==============================================================================
# 配置文件路径（固定为你指定的路径）
# ==============================================================================
CONFIG_FILE_PATH = r'D:\PycharmProjects\PythonProject\YOLO参数.yaml'

# ==============================================================================
# 读取配置文件函数
# ==============================================================================
def load_config(config_path):
    """读取YAML配置文件，返回配置字典（不存在则返回默认配置）"""
    # 默认配置（与你原代码一致，作为 fallback）
    default_config = {
        # 设备配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_mem_clear': True,
        'cudnn_benchmark': True,
        'allow_tf32': True,
        # 屏幕截取配置
        'screen_width': 2560,
        'screen_height': 1600,
        'window_left': 2560 // 3,
        'window_top': 1600 // 3,
        'window_width': 2560 // 3,
        'window_height': 1600 // 3,
        'window_name': 'YOLO_CS2',
        'model_size': 640,
        # 模型参数
        'target_class_ids': [0, 1],
        'conf_threshold': 0.5,
        'iou_threshold': 0.4,
        # 保存功能配置
        'save_dir': 'yolo_detection_saves',
        'save_key': 's',
        'exit_key': 27,  # ESC键ASCII码
        # 显示配置
        'fps_display': True,
        'fps_position': [10, 25],
        'fps_font_size': 0.5,
        'fps_color': [0, 255, 0],  # BGR绿色
        'fps_line_width': 1,
        'detect_line_width': 1
    }

    # 读取配置文件，覆盖默认值
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            # 逐个覆盖默认配置（只覆盖用户指定的键）
            for key, value in user_config.items():
                if key in default_config:
                    default_config[key] = value
            print(f"✅ 成功读取配置文件：{config_path}")
        except Exception as e:
            print(f"❌ 配置文件读取失败：{str(e)}，使用默认配置")
    else:
        print(f"⚠️  配置文件不存在：{config_path}，使用默认配置并自动生成配置文件")
        # 自动生成默认配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True, sort_keys=False)
        print(f"✅ 已自动生成默认配置文件：{config_path}")

    return default_config

# ==============================================================================
# 加载配置文件
# ==============================================================================
config = load_config(CONFIG_FILE_PATH)

# ==============================================================================
# 设备初始化（基于配置文件）
# ==============================================================================
device = config['device']
print(f'Device: {device}')
if device == 'cuda' and torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    if config['gpu_mem_clear']:
        torch.cuda.empty_cache()
    if config['cudnn_benchmark']:
        torch.backends.cudnn.benchmark = True
    if config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
elif device == 'cuda' and not torch.cuda.is_available():
    print("⚠️  无GPU，自动切换为CPU")
    device = 'cpu'

# ==============================================================================
# 屏幕截取配置（基于配置文件）
# ==============================================================================
screen_width = config['screen_width']
screen_height = config['screen_height']
WINDOW_LEFT = config['window_left']
WINDOW_TOP = config['window_top']
WINDOW_WIDTH = config['window_width']
WINDOW_HEIGHT = config['window_height']
window_name = config['window_name']
model_size = config['model_size']

monitor = {
    'left': WINDOW_LEFT,
    'top': WINDOW_TOP,
    'width': WINDOW_WIDTH,
    'height': WINDOW_HEIGHT
}

# ==============================================================================
# 保存功能配置（基于配置文件）
# ==============================================================================
save_dir = config['save_dir']
save_key = ord(config['save_key'])  # 转换为ASCII码
exit_key = config['exit_key']
os.makedirs(save_dir, exist_ok=True)

# ==============================================================================
# 模型参数配置（基于配置文件）
# ==============================================================================
target_class_ids = config['target_class_ids']
conf_threshold = config['conf_threshold']
iou_threshold = config['iou_threshold']
detect_line_width = config['detect_line_width']

# ==============================================================================
# 显示配置（基于配置文件）
# ==============================================================================
fps_display = config['fps_display']
fps_position = tuple(config['fps_position'])
fps_font_size = config['fps_font_size']
fps_color = tuple(config['fps_color'])
fps_line_width = config['fps_line_width']

# ==============================================================================
# 加载模型（基于配置文件）
# ==============================================================================
model = YOLO('cs2.pt').to(device)
model.fuse()  # 融合层提速，不影响稳定性

# ==============================================================================
# 帧率计算初始化
# ==============================================================================
fps_start_time = time.time()
fps_count = 0
fps_text = "0 FPS"  # 初始值，避免报错

# ==============================================================================
# 预处理函数
# ==============================================================================
def preprocess_image(img_np):
    """保留你的预处理逻辑，仅优化传输速度"""
    img_tensor = torch.from_numpy(img_np).float().to(device, non_blocking=True)  # 异步传输
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = torch.nn.functional.interpolate(
        img_tensor,
        size=(model_size, model_size),
        mode='bilinear',
        align_corners=False
    )
    return img_tensor

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    sct = mss.mss()
    # 保留你的简洁窗口创建逻辑
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. 截取屏幕（保留你的方式）
            img = np.array(sct.grab(monitor))[:, :, :3]

            # 2. 计算帧率（基于配置文件开关）
            if fps_display:
                fps_count += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = int(fps_count / (time.time() - fps_start_time))
                    fps_text = f"{fps} FPS"
                    fps_count = 0
                    fps_start_time = time.time()

            # 3. 预处理+推理（参数来自配置文件）
            img_tensor = preprocess_image(img)
            with torch.no_grad():
                results = model(
                    img_tensor,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    classes=target_class_ids
                )

            # 4. 绘制检测框（线宽来自配置文件）
            annotated_frame = results[0].plot(line_width=detect_line_width)
            annotated_frame = cv2.resize(annotated_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

            # 5. 绘制帧率（基于配置文件）
            if fps_display:
                cv2.putText(
                    annotated_frame,
                    fps_text,
                    fps_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fps_font_size,
                    fps_color,
                    fps_line_width,
                    cv2.LINE_AA
                )

            # 6. 显示窗口+置顶（保留你的逻辑）
            cv2.imshow(window_name, annotated_frame)
            hwnd = win32gui.FindWindow(None, window_name)
            win32gui.SetWindowPos(
                hwnd, win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOSIZE
            )

            # 7. 按键操作（按键来自配置文件）
            key = cv2.waitKey(1) & 0xFF
            if key == exit_key:
                break
            elif key == save_key:
                save_path = os.path.join(save_dir, f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(save_path, annotated_frame)
                print(f"已保存至：{save_path}")

    finally:
        # 保留你的资源清理逻辑
        cv2.destroyAllWindows()
        sct.close()
        if device == 'cuda' and config['gpu_mem_clear']:
            torch.cuda.empty_cache()
