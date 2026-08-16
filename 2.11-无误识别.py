# -*- coding: utf-8 -*-
'''
Script: deploy_det_video.py
脚本名称：deploy_det_video.py

Description:
    This script runs a real-time object detection application on an embedded device.
    It uses a pipeline to capture video frames, performs inference using a pre-trained Kmodel,
    and displays the detection results (bounding boxes, class labels) on screen.

    The model configuration is loaded from the Canaan online training platform via a JSON config file.

脚本说明：
    本脚本在嵌入式设备上运行实时目标检测应用。它通过捕获视频帧，使用预训练的 Kmodel 进行推理，并在屏幕上显示检测结果（边界框、类别标签）。

    模型配置文件通过 Canaan 在线训练平台从 JSON 文件加载。

Author: Canaan Developer
作者：Canaan 开发者
'''


import os, gc
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

# Set display mode: options are 'hdmi', 'lcd', 'lt9611', 'st7701', 'hx8399'
# 'hdmi' defaults to 'lt9611' (1920x1080); 'lcd' defaults to 'st7701' (800x480)
display_mode = "st7701"

# Define the input size for the RGB888P video frames
rgb888p_size = [1280, 720]

# Set root directory path for model and config
root_path = "/sdcard/mp_deployment_source/"

# Load deployment configuration
deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]              # KModel path
labels = deploy_conf["categories"]                                # Label list
confidence_threshold = deploy_conf["confidence_threshold"]        # Confidence threshold
nms_threshold = deploy_conf["nms_threshold"]                      # NMS threshold
model_input_size = deploy_conf["img_size"]                        # Model input size
nms_option = deploy_conf["nms_option"]                            # NMS strategy
model_type = deploy_conf["model_type"]                            # Detection model type
anchors = []
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

# Inference configuration
inference_mode = "video"                                          # Inference mode: 'video'
debug_mode = 0                                                    # Debug mode flag

# Create and initialize the video/display pipeline
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

# ========== 坐标转换计算 ==========
# 原始图像分辨率 vs 显示分辨率
src_w, src_h = rgb888p_size[0], rgb888p_size[1]
dst_w, dst_h = display_size[0], display_size[1]
scale_x = dst_w / src_w  # X轴缩放比例
scale_y = dst_h / src_h  # Y轴缩放比例
print(f"[坐标] 原始:{src_w}x{src_h} -> 显示:{dst_w}x{dst_h}")
print(f"[坐标] 缩放比例: X={scale_x:.3f}, Y={scale_y:.3f}")

def coord_to_display(x, y, w, h):
    """将原始图像坐标转换为显示坐标"""
    return int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

# Initialize object detection application
det_app = DetectionApp(inference_mode,kmodel_path,labels,model_input_size,anchors,model_type,confidence_threshold,nms_threshold,rgb888p_size,display_size,debug_mode=debug_mode)

# Configure preprocessing for the model
det_app.config_preprocess()

# ========== Tracker 配置参数 ==========
# 解决小目标闪烁问题：连续检测 CONFIRM_HITS 次才确认，丢帧 COAST_MAX 次仍保持跟踪
CONFIRM_HITS = 2     # 连续检测几次才算确认目标
COAST_MAX = 3        # 允许连续丢帧几次仍保持跟踪（减小，更快消失）
MATCH_DIST = 100     # 匹配距离阈值（像素），增大更容易匹配
EMA_ALPHA = 0.7      # 位置平滑系数：增大，响应更快跟随当前检测

# ========== Tracker 类定义 ==========
class Track:
    """单个目标的跟踪状态"""
    __slots__ = ("cx", "cy", "w", "h", "score", "hits", "misses", "confirmed")

    def __init__(self, cx, cy, w, h, score):
        self.cx, self.cy, self.w, self.h, self.score = cx, cy, w, h, score
        self.hits, self.misses, self.confirmed = 1, 0, False


class Tracker:
    """多目标跟踪器，带 coasting 功能（丢帧时保持显示）"""

    def __init__(self):
        self.tracks = []

    def update(self, dets):
        """
        输入：dets = [(cx, cy, w, h, score), ...]
        输出：确认的稳定目标列表
        """
        used = [False] * len(dets)
        d2max = MATCH_DIST * MATCH_DIST

        # 1) 匹配现有轨迹到最近的检测结果
        for t in self.tracks:
            best, bestd = -1, d2max
            for i, d in enumerate(dets):
                if used[i]:
                    continue
                dd = (d[0] - t.cx) ** 2 + (d[1] - t.cy) ** 2
                if dd < bestd:
                    bestd, best = dd, i
            if best >= 0:
                d = dets[best]
                used[best] = True
                a = EMA_ALPHA
                t.cx = a * d[0] + (1 - a) * t.cx
                t.cy = a * d[1] + (1 - a) * t.cy
                t.w = a * d[2] + (1 - a) * t.w
                t.h = a * d[3] + (1 - a) * t.h
                t.score = d[4]
                t.hits += 1
                t.misses = 0
                if t.hits >= CONFIRM_HITS:
                    t.confirmed = True
            else:
                t.misses += 1  # coasting：保持位置，计数丢失帧

        # 2) 为未匹配的检测创建新轨迹
        for i, d in enumerate(dets):
            if not used[i]:
                self.tracks.append(Track(*d))

        # 3) 移除丢失太久的目标
        self.tracks = [t for t in self.tracks if t.misses <= COAST_MAX]

        # 4) 返回确认的稳定目标（包括正在coasting的）
        return [t for t in self.tracks if t.confirmed]


# ========== 辅助函数：格式转换 ==========
def res_to_dets(res):
    """
    将 Canaan SDK 的 res (dict) 转换为 Tracker 需要的格式
    输入：res = {'scores': array, 'idx': array, 'boxes': array}
    输出：dets = [(cx, cy, w, h, score), ...]
    """
    if res is None or not isinstance(res, dict):
        return []
    dets = []
    scores = res.get('scores', [])
    boxes = res.get('boxes', [])
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        dets.append((
            (x1 + x2) / 2,  # cx
            (y1 + y2) / 2,  # cy
            x2 - x1,        # w
            y2 - y1,        # h
            float(scores[i])
        ))
    return dets


# ---------- 帧计数器和 FPS 计算 ----------
frame_cnt = 0
PRINT_INTERVAL = 30   # 每30帧打印一次检测状态
tracker = Tracker()   # 初始化跟踪器

# FPS 计算（需要 time 模块）
try:
    import time
    fps_t = time.ticks_ms()
    FPS_ENABLED = True
except:
    FPS_ENABLED = False

# Main loop: capture, run inference, display results
while True:
    with ScopedTiming("total", 1):
        img = pl.get_frame()                          # Capture current frame
        res = det_app.run(img)                        # Run inference

        # ---------- Tracker 集成：解决闪烁问题 ----------
        dets = res_to_dets(res)                       # 转换格式
        stable = tracker.update(dets)                 # 获取稳定跟踪目标

        # ---------- 检测状态打印 + 坐标对比调试 ----------
        frame_cnt += 1
        if frame_cnt % PRINT_INTERVAL == 0:
            num_det = len(dets)                       # 原始检测数
            num_track = len(stable)                   # 稳定跟踪数
            status = "✅" if num_track > 0 else "❌"
            print(f"[帧 {frame_cnt}] {status} 检测:{num_det} 跟踪:{num_track}")

            # 打印原始 boxes 坐标（红框位置）
            if res is not None and isinstance(res, dict) and len(res.get('scores', [])) > 0:
                boxes = res['boxes']
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    w = x2 - x1
                    h = y2 - y1
                    print(f"  红框[{i}]: boxes=[{x1},{y1},{x2},{y2}] -> 中心=({cx},{cy}) 尺寸=({w},{h})")

            # 打印 Tracker 输出的坐标（绿框位置）
            for i, t in enumerate(stable):
                x = int(t.cx - t.w / 2)
                y = int(t.cy - t.h / 2)
                print(f"  绿框[{i}]: tracker -> 中心=({int(t.cx)},{int(t.cy)}) 尺寸=({int(t.w)},{int(t.h)}) 绘制起点=({x},{y})")

        # ---------- 绘制 ----------
        det_app.draw_result(pl.osd_img, res)          # 原始红框（保留）

        # 绘制稳定跟踪的绿框（使用坐标转换）
        for t in stable:
            x = int(t.cx - t.w / 2)
            y = int(t.cy - t.h / 2)
            cx, cy = int(t.cx), int(t.cy)
            w, h = int(t.w), int(t.h)

            # 转换到显示坐标
            dx, dy, dw, dh = coord_to_display(x, y, w, h)
            dcx, dcy = int(cx * scale_x), int(cy * scale_y)

            try:
                # 绿色：稳定跟踪框（转换后坐标）
                pl.osd_img.draw_rectangle(dx, dy, dw, dh, (0, 255, 0), 2)
                # 红色十字：目标中心（转换后坐标）
                pl.osd_img.draw_cross(dcx, dcy, (255, 0, 0), 8, 2)
            except:
                pass

        # ---------- FPS 显示 ----------
        if FPS_ENABLED and frame_cnt % PRINT_INTERVAL == 0:
            now = time.ticks_ms()
            fps = PRINT_INTERVAL * 1000.0 / max(1, now - fps_t)
            print(f"  FPS: {fps:.1f}")
            fps_t = now
            # 尝试在屏幕上显示FPS（如果支持）
            try:
                pl.osd_img.draw_string(4, 4, f"FPS:{fps:.1f}", (255, 255, 255), 1)
            except:
                pass

        pl.show_image()                               # Show result on disp
        gc.collect()                                  # Run garbage collection
# Cleanup: These lines will only run if the loop is interrupted (e.g., by an IDE break or external interruption)
det_app.deinit()                                      # De-initialize detection app
pl.destroy()                                          # Destroy pipeline instance
