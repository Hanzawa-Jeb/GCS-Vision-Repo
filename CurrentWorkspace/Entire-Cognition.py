from maix import camera, display, image, nn, app
import time

# detector = nn.YOLO11(model="/root/ObjCogModel/yolov11n.mud", dual_buff = True) # 识别柱子
detector = nn.YOLO11(model="/root/VisionVer1/yolov11n.mud", dual_buff = True)   # 识别颜色环

cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()

# 初始化计时
last_time = time.time()
fps = 0.0

while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th=0.5, iou_th=0.45)

    # 绘制检测结果
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)

    # 计算 FPS
    now = time.time()
    dt = now - last_time   # 秒
    if dt > 0:
        fps = 1.0 / dt
    last_time = now

    # 在右上角显示 FPS
    fps_text = f"FPS: {fps:.1f}"
    img.draw_string(img.width() - 100, 5, fps_text, color=image.COLOR_GREEN)

    disp.show(img)
