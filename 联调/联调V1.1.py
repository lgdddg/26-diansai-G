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
import ustruct
from machine import UART
from machine import FPIOA
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

# ===================== 串口初始化 =====================
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.UART2_TXD)
fpioa.set_function(12, FPIOA.UART2_RXD)
uart = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

def send_distance(dist_cm):
    """
    发送距离数据到串口
    帧结构：0x2C + 0x5A + 数据字节 + 0xFE
    数据字节编码：最高位=符号位(0=正/左,1=负/右)，低7位=距离×10
    例如：+2.8cm -> 0x1C(28), -2.8cm -> 0x9C(28|0x80)
    """
    dist_x10 = int(abs(dist_cm) * 10)  # 距离×10
    sign_bit = 0 if dist_cm >= 0 else 0x80  # 符号位
    data_byte = sign_bit | dist_x10
    frame = ustruct.pack(">BBBB", 0x2C, 0x5A, data_byte, 0xFE)
    uart.write(frame)
    # 打印发送信息
    sign_str = "+" if dist_cm >= 0 else "-"
    print(f"[UART] 距离={sign_str}{abs(dist_cm)}cm, 发送: {' '.join('0x{:02X}'.format(b) for b in frame)}")
# ========================================================

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

# ========== 标定常量（像素转厘米） ==========
X0 = 649.0          # 中心像素点
X_LEFT5 = 892.0     # 左5cm点
X_RIGHT5 = 326.0    # 右5cm点
K_LEFT  = 5.0 / (X_LEFT5 - X0)   # 左侧系数 cm/像素
K_RIGHT = -5.0 / (X_RIGHT5 - X0) # 右侧系数 cm/像素

def pixel_to_cm(target_x):
    """
    输入：图像x像素（浮点/整数均可）
    返回：带正负距离，保留1位小数（单位cm）
    > 正数 = 中心点左侧
    > 负数 = 中心点右侧
    """
    if target_x >= X0:
        dist = (target_x - X0) * K_LEFT
    else:
        dist = (target_x - X0) * K_RIGHT
    return round(dist, 1)

# ========== ROI 区域定义 ==========
# 在显示坐标系下的ROI: (x:46, y:132, w:693, h:73)
ROI_DISPLAY = (3, 183, 792, 68)

# 将显示坐标的ROI转换到原始图像坐标
roi_x_src = int(ROI_DISPLAY[0] / scale_x)
roi_y_src = int(ROI_DISPLAY[1] / scale_y)
roi_w_src = int(ROI_DISPLAY[2] / scale_x)
roi_h_src = int(ROI_DISPLAY[3] / scale_y)
ROI_SRC = (roi_x_src, roi_y_src, roi_w_src, roi_h_src)
print(f"[ROI] 显示坐标: {ROI_DISPLAY}")
print(f"[ROI] 原始坐标: {ROI_SRC}")

def is_in_roi(x, y, w, h):
    """检查检测框是否与ROI有重叠"""
    roi_x, roi_y, roi_w, roi_h = ROI_SRC
    # 检测框的中心
    cx, cy = x + w // 2, y + h // 2
    # 检查中心是否在ROI内（有一定容差）
    return (roi_x <= cx <= roi_x + roi_w and
            roi_y <= cy <= roi_y + roi_h)

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
    将 Canaan SDK 的 res (dict) 转换为 Tracker 需要的格式，并过滤ROI外的检测
    输入：res = {'scores': array, 'idx': array, 'boxes': array}
    输出：dets = [(cx, cy, w, h, score), ...]  (仅包含ROI内的检测)
    """
    if res is None or not isinstance(res, dict):
        return []
    dets = []
    scores = res.get('scores', [])
    boxes = res.get('boxes', [])
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # 仅保留ROI内的检测
        if is_in_roi(int(x1), int(y1), int(w), int(h)):
            dets.append((cx, cy, w, h, float(scores[i])))
    return dets

def res_to_all_dets(res):
    """
    将 Canaan SDK 的 res 转换为所有检测（不过滤ROI），用于绘制原始红框
    """
    if res is None or not isinstance(res, dict):
        return []
    dets = []
    scores = res.get('scores', [])
    boxes = res.get('boxes', [])
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        dets.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(scores[i])))
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
    with ScopedTiming("total", 0):
        img = pl.get_frame()                          # Capture current frame
        res = det_app.run(img)                        # Run inference

        # ---------- Tracker 集成：解决闪烁问题 ----------
        dets = res_to_dets(res)                       # 转换格式
        stable = tracker.update(dets)                 # 获取稳定跟踪目标

        # ---------- 串口发送距离数据 ----------
        if len(stable) > 0:
            dist_cm = pixel_to_cm(int(stable[0].cx))  # 取第一个稳定目标的距离
            send_distance(dist_cm)

        # ---------- 检测状态打印 + 坐标对比调试 ----------
        frame_cnt += 1
        if frame_cnt % PRINT_INTERVAL == 0:
            num_det = len(dets)                       # ROI内检测数
            num_track = len(stable)                   # 稳定跟踪数
            # 统计总检测数和ROI外数量
            total_det = len(res.get('scores', [])) if res else 0
            roi_out_det = total_det - num_det         # ROI外被过滤的数量
            status = "✅" if num_track > 0 else "❌"
            print(f"[帧 {frame_cnt}] {status} ROI内:{num_det} ROI外:{roi_out_det} 跟踪:{num_track}")

            # # 打印所有检测的位置（标注ROI内外）
            # if res is not None and isinstance(res, dict) and len(res.get('scores', [])) > 0:
            #     boxes = res['boxes']
            #     for i in range(len(boxes)):
            #         x1, y1, x2, y2 = boxes[i]
            #         cx = (x1 + x2) // 2
            #         cy = (y1 + y2) // 2
            #         w = x2 - x1
            #         h = y2 - y1
            #         in_roi = is_in_roi(x1, y1, w, h)
            #         tag = "[ROI内]" if in_roi else "[ROI外]"
            #         dist_cm = pixel_to_cm(cx)
            #         side = "左侧" if dist_cm >= 0 else "右侧"
            #         print(f"  {tag}[{i}]: boxes=[{x1},{y1},{x2},{y2}] -> 中心=({cx},{cy}) 尺寸=({w},{h}) 距离={abs(dist_cm)}cm({side})")

            # 打印 Tracker 输出的坐标（绿框位置）
            for i, t in enumerate(stable):
                x = int(t.cx - t.w / 2)
                y = int(t.cy - t.h / 2)
                dist_cm = pixel_to_cm(int(t.cx))
                side = "左侧" if dist_cm >= 0 else "右侧"
                print(f"  绿框[{i}]: tracker -> 中心=({int(t.cx)},{int(t.cy)}) 尺寸=({int(t.w)},{int(t.h)}) 距离={abs(dist_cm)}cm({side}) 绘制起点=({x},{y})")

        # ---------- 绘制 ----------
        # 先清空OSD图像（清除上一帧的所有框）
        try:
            pl.osd_img.clear()
        except:
            pass

        # 绘制ROI区域框（青色边框）
        try:
            rx, ry, rw, rh = ROI_DISPLAY
            pl.osd_img.draw_rectangle(rx, ry, rw, rh, (0, 255, 255), 2)  # 青色ROI框
        except:
            pass

        # 绘制原始红框（仅ROI内的检测用红色）
        all_dets = res_to_all_dets(res)
        for d in all_dets:
            cx, cy, w, h, score = d
            x1, y1 = int(cx - w/2), int(cy - h/2)
            if is_in_roi(x1, y1, int(w), int(h)):
                dx, dy, dw, dh = coord_to_display(x1, y1, int(w), int(h))
                try:
                    pl.osd_img.draw_rectangle(dx, dy, dw, dh, (255, 0, 0), 2)  # 红色：ROI内
                except:
                    pass

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
uart.deinit()                                         # 释放串口
det_app.deinit()                                      # De-initialize detection app
pl.destroy()                                          # Destroy pipeline instance
