
# -*- coding: utf-8 -*-
'''
脚本：联调V1.3（识别修改绿框）
基于：联调V1.2（修改补码版）

主要改进：
    1. 检测去重（本地NMS）：喂给Tracker前，对AI输出的重叠检测框去重
    2. 速度预测匹配：Tracker根据目标历史速度预测下一帧位置，解决高速小球丢失
    3. Coasting滑行：目标丢失时位置按惯性继续滑动，而非冻结
    4. 邻近抑制：新建轨迹前检查是否离已有轨迹太近，避免重复轨迹
    5. 轨迹间NMS：返回confirmed前，对重叠轨迹去重
    6. 单目标硬限制：最终输出只保留置信度最高的1个目标，杜绝双绿框
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
    发送距离数据到串口（补码编码，与V1.2一致）
    """
    dist_x10 = int(round(dist_cm * 10))
    if dist_x10 > 127:
        dist_x10 = 127
    elif dist_x10 < -128:
        dist_x10 = -128
    data_byte = dist_x10 & 0xFF
    frame = ustruct.pack(">BBBB", 0x2C, 0x5A, data_byte, 0xFE)
    uart.write(frame)
# ========================================================

# Set display mode
display_mode = "st7701"

# Define the input size for the RGB888P video frames
rgb888p_size = [1280, 720]

# Set root directory path for model and config
root_path = "/sdcard/mp_deployment_source/"

# Load deployment configuration
deploy_conf = read_json(root_path + "/deploy_config.json")
kmodel_path = root_path + deploy_conf["kmodel_path"]
labels = deploy_conf["categories"]
confidence_threshold = deploy_conf["confidence_threshold"]
nms_threshold = deploy_conf["nms_threshold"]
model_input_size = deploy_conf["img_size"]
nms_option = deploy_conf["nms_option"]
model_type = deploy_conf["model_type"]
anchors = []
if model_type == "AnchorBaseDet":
    anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]

# Inference configuration
inference_mode = "video"
debug_mode = 0

# Create and initialize the video/display pipeline
pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode)
pl.create()
display_size = pl.get_display_size()

# ========== 坐标转换计算 ==========
src_w, src_h = rgb888p_size[0], rgb888p_size[1]
dst_w, dst_h = display_size[0], display_size[1]
scale_x = dst_w / src_w
scale_y = dst_h / src_h
print(f"[坐标] 原始:{src_w}x{src_h} -> 显示:{dst_w}x{dst_h}")
print(f"[坐标] 缩放比例: X={scale_x:.3f}, Y={scale_y:.3f}")

def coord_to_display(x, y, w, h):
    """将原始图像坐标转换为显示坐标"""
    return int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

# ========== 标定常量（像素转厘米） ==========
X0 = 547.0          # 中心像素点
X_LEFT5 = 770.0     # 左5cm点
X_RIGHT5 = 289.0    # 右5cm点
K_LEFT  = 5.0 / (X_LEFT5 - X0)   # 左侧系数 cm/像素
K_RIGHT = -5.0 / (X_RIGHT5 - X0) # 右侧系数 cm/像素

def pixel_to_cm(target_x):
    """
    输入：图像x像素
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
ROI_DISPLAY = (0, 175, 745, 44)

roi_x_src = int(ROI_DISPLAY[0] / scale_x)
roi_y_src = int(ROI_DISPLAY[1] / scale_y)
roi_w_src = int(ROI_DISPLAY[2] / scale_x)
roi_h_src = int(ROI_DISPLAY[3] / scale_y)
ROI_SRC = (roi_x_src, roi_y_src, roi_w_src, roi_h_src)
print(f"[ROI] 显示坐标: {ROI_DISPLAY}")
print(f"[ROI] 原始坐标: {ROI_SRC}")

def is_in_roi(x, y, w, h):
    """检查检测框中心是否在ROI内"""
    roi_x, roi_y, roi_w, roi_h = ROI_SRC
    cx, cy = x + w // 2, y + h // 2
    return (roi_x <= cx <= roi_x + roi_w and
            roi_y <= cy <= roi_y + roi_h)

# Initialize object detection application
det_app = DetectionApp(inference_mode,kmodel_path,labels,model_input_size,anchors,model_type,confidence_threshold,nms_threshold,rgb888p_size,display_size,debug_mode=debug_mode)
det_app.config_preprocess()

# ========== 检测去重（本地NMS） ==========
DEDUP_DIST = 30  # 中心距离小于此值（像素）的检测框视为同一物体

def dedup_dets(dets):
    """
    对AI输出的检测框做本地去重。
    如果两个检测框中心距离 < DEDUP_DIST，只保留置信度更高的那个。
    解决AI对同一小球输出多个重叠框的问题。
    """
    if len(dets) <= 1:
        return dets
    # 按置信度降序排序
    sorted_dets = sorted(dets, key=lambda d: d[4], reverse=True)
    kept = []
    for d in sorted_dets:
        is_dup = False
        for k in kept:
            dx = d[0] - k[0]
            dy = d[1] - k[1]
            if dx * dx + dy * dy < DEDUP_DIST * DEDUP_DIST:
                is_dup = True
                break
        if not is_dup:
            kept.append(d)
    return kept

# ========== Tracker 配置参数 ==========
CONFIRM_HITS = 2     # 连续检测几次才算确认目标
COAST_MAX = 5        # 允许连续丢帧仍保持跟踪（增大，配合速度预测滑行）
MATCH_DIST = 150     # 匹配距离阈值（像素），增大配合预测位置
EMA_ALPHA = 0.6      # 位置平滑系数
VEL_ALPHA = 0.5      # 速度平滑系数
MAX_TRACKS = 1       # 单目标场景：最多保留1条confirmed轨迹

# ========== Tracker 类定义（带速度预测） ==========
class Track:
    """单个目标的跟踪状态，含速度属性"""
    __slots__ = ("cx", "cy", "w", "h", "score", "hits", "misses", "confirmed", "vx", "vy")

    def __init__(self, cx, cy, w, h, score):
        self.cx, self.cy, self.w, self.h, self.score = cx, cy, w, h, score
        self.hits, self.misses, self.confirmed = 1, 0, False
        self.vx, self.vy = 0.0, 0.0  # 初始速度为0


class Tracker:
    """多目标跟踪器，带速度预测 + coasting滑行 + 轨迹NMS"""

    def __init__(self):
        self.tracks = []

    def update(self, dets):
        """
        输入：dets = [(cx, cy, w, h, score), ...]
        输出：确认的稳定目标列表（已去重，最多MAX_TRACKS个）
        """
        used = [False] * len(dets)
        d2max = MATCH_DIST * MATCH_DIST

        # 1) 匹配：用"预测位置"匹配检测结果（而非上一帧位置）
        for t in self.tracks:
            # 速度预测：下一帧位置 = 当前位置 + 速度
            pred_cx = t.cx + t.vx
            pred_cy = t.cy + t.vy

            best, bestd = -1, d2max
            for i, d in enumerate(dets):
                if used[i]:
                    continue
                # 用预测位置计算距离
                dd = (d[0] - pred_cx) ** 2 + (d[1] - pred_cy) ** 2
                if dd < bestd:
                    bestd, best = dd, i
            if best >= 0:
                d = dets[best]
                used[best] = True
                a = EMA_ALPHA
                # 位置EMA平滑
                new_cx = a * d[0] + (1 - a) * t.cx
                new_cy = a * d[1] + (1 - a) * t.cy
                # 计算本次位移作为速度观测值
                obs_vx = new_cx - t.cx
                obs_vy = new_cy - t.cy
                # 速度EMA平滑
                va = VEL_ALPHA
                t.vx = va * obs_vx + (1 - va) * t.vx
                t.vy = va * obs_vy + (1 - va) * t.vy
                # 更新位置
                t.cx, t.cy = new_cx, new_cy
                t.w = a * d[2] + (1 - a) * t.w
                t.h = a * d[3] + (1 - a) * t.h
                t.score = d[4]
                t.hits += 1
                t.misses = 0
                if t.hits >= CONFIRM_HITS:
                    t.confirmed = True
            else:
                # coasting：位置按速度继续滑行（而非冻结）
                t.cx += t.vx
                t.cy += t.vy
                t.misses += 1

        # 2) 为未匹配的检测创建新轨迹（带邻近抑制）
        for i, d in enumerate(dets):
            if not used[i]:
                # 邻近抑制：如果这个检测离任何已有轨迹太近，丢弃它
                too_close = False
                for t in self.tracks:
                    dx = d[0] - t.cx
                    dy = d[1] - t.cy
                    if dx * dx + dy * dy < DEDUP_DIST * DEDUP_DIST:
                        too_close = True
                        break
                if not too_close:
                    self.tracks.append(Track(*d))

        # 3) 移除丢失太久的目标
        self.tracks = [t for t in self.tracks if t.misses <= COAST_MAX]

        # 4) 轨迹间NMS：去掉重叠的confirmed轨迹
        confirmed = [t for t in self.tracks if t.confirmed]
        confirmed.sort(key=lambda t: t.score, reverse=True)
        kept = []
        for t in confirmed:
            is_dup = False
            for k in kept:
                dx = t.cx - k.cx
                dy = t.cy - k.cy
                if dx * dx + dy * dy < DEDUP_DIST * DEDUP_DIST:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(t)

        # 5) 单目标硬限制：只保留置信度最高的MAX_TRACKS个
        return kept[:MAX_TRACKS]


# ========== 辅助函数：格式转换 ==========
def res_to_dets(res):
    """
    将 Canaan SDK 的 res 转换为 Tracker 需要的格式，并过滤ROI外的检测
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
        if is_in_roi(int(x1), int(y1), int(w), int(h)):
            dets.append((cx, cy, w, h, float(scores[i])))
    return dets

def res_to_all_dets(res):
    """将 res 转换为所有检测（不过滤ROI），用于绘制原始红框"""
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
PRINT_INTERVAL = 30
tracker = Tracker()

try:
    import time
    fps_t = time.ticks_ms()
    FPS_ENABLED = True
except:
    FPS_ENABLED = False

# Main loop: capture, run inference, display results
while True:
    with ScopedTiming("total", 0):
        img = pl.get_frame()
        res = det_app.run(img)

        # ---------- 检测去重 → Tracker ----------
        raw_dets = res_to_dets(res)             # ROI内的原始检测
        dets = dedup_dets(raw_dets)             # 去重后的检测
        stable = tracker.update(dets)           # 获取稳定跟踪目标

        # ---------- 串口发送距离数据 ----------
        if len(stable) > 0:
            dist_cm = pixel_to_cm(int(stable[0].cx))
            send_distance(dist_cm)

        # ---------- 检测状态打印 ----------
        frame_cnt += 1
        if frame_cnt % PRINT_INTERVAL == 0:
            num_raw = len(raw_dets)
            num_dedup = len(dets)
            num_track = len(stable)
            total_det = len(res.get('scores', [])) if res else 0
            status = "✅" if num_track > 0 else "❌"
            print(f"[帧 {frame_cnt}] {status} 总检测:{total_det} ROI内:{num_raw} 去重后:{num_dedup} 跟踪:{num_track}")

            # 打印 Tracker 输出的坐标（绿框位置）
            for i, t in enumerate(stable):
                x = int(t.cx - t.w / 2)
                y = int(t.cy - t.h / 2)
                dist_cm = pixel_to_cm(int(t.cx))
                side = "左侧" if dist_cm >= 0 else "右侧"
                print(f"  绿框[{i}]: tracker -> 中心=({int(t.cx)},{int(t.cy)}) 尺寸=({int(t.w)},{int(t.h)}) 速度=({t.vx:.1f},{t.vy:.1f}) 距离={abs(dist_cm)}cm({side}) 绘制起点=({x},{y})")

        # ---------- 绘制 ----------
        try:
            pl.osd_img.clear()
        except:
            pass

        # 绘制ROI区域框（青色边框）
        try:
            rx, ry, rw, rh = ROI_DISPLAY
            pl.osd_img.draw_rectangle(rx, ry, rw, rh, (0, 255, 255), 2)
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
                    pl.osd_img.draw_rectangle(dx, dy, dw, dh, (255, 0, 0), 2)
                except:
                    pass

        # 绘制稳定跟踪的绿框
        for t in stable:
            x = int(t.cx - t.w / 2)
            y = int(t.cy - t.h / 2)
            cx, cy = int(t.cx), int(t.cy)
            w, h = int(t.w), int(t.h)

            dx, dy, dw, dh = coord_to_display(x, y, w, h)
            dcx, dcy = int(cx * scale_x), int(cy * scale_y)

            try:
                pl.osd_img.draw_rectangle(dx, dy, dw, dh, (0, 255, 0), 2)
                pl.osd_img.draw_cross(dcx, dcy, (255, 0, 0), 8, 2)
            except:
                pass

        # ---------- FPS 显示 ----------
        if FPS_ENABLED and frame_cnt % PRINT_INTERVAL == 0:
            now = time.ticks_ms()
            fps = PRINT_INTERVAL * 1000.0 / max(1, now - fps_t)
            print(f"  FPS: {fps:.1f}")
            fps_t = now
            try:
                pl.osd_img.draw_string(4, 4, f"FPS:{fps:.1f}", (255, 255, 255), 1)
            except:
                pass

        pl.show_image()
        gc.collect()

# Cleanup
uart.deinit()
det_app.deinit()
pl.destroy()
