from maix import camera, display, image, nn, app
import time
import math

# 初始化YOLOv11模型，用于识别颜色环或柱子
detector = nn.YOLO11(model="/root/VisionVer1/yolov11n.mud", dual_buff=True)

# 初始化摄像头，尺寸与YOLO模型输入尺寸一致
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

# 初始化计时
last_time = time.time()
fps = 0.0

while not app.need_exit():
    # 读取摄像头图像
    img = cam.read()

    # --- 目标检测部分 (识别颜色环/柱子) ---
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)
    
    # 在图像上绘制目标检测结果
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED, thickness=2)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED, scale=1.5)

    # --- 直线检测部分 ---
    # `threshold` 值可能需要根据光照和图像内容进行调整
    lines = img.find_lines(threshold=3000)
    for line in lines:
        # 在图像上绘制检测到的直线
        img.draw_line(line.x1(), line.y1(), line.x2(), line.y2(), color=image.COLOR_GREEN, thickness=2)
        
        # 绘制直线的极坐标信息（可选，可删除）
        theta = line.theta()
        rho = line.rho()
        angle_in_radians = math.radians(theta)
        x_text = int(math.cos(angle_in_radians) * rho)
        y_text = int(math.sin(angle_in_radians) * rho)
        img.draw_string(x_text, y_text, f"theta: {theta}, rho: {rho}", color=image.COLOR_GREEN, scale=1.5)

    # --- FPS计算与显示 ---
    now = time.time()
    dt = now - last_time
    if dt > 0:
        fps = 1.0 / dt
    last_time = now

    fps_text = f"FPS: {fps:.1f}"
    # 在右上角显示FPS
    img.draw_string(img.width() - 150, 5, fps_text, color=image.COLOR_WHITE, scale=2)

    # 在显示器上显示处理后的图像
    disp.show(img)