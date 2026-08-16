
# -*- coding: utf-8 -*-
'''
脚本：联调V2.1（按键选取位置）
基于：联调V1.3（识别修改绿框）

V2.1 新增功能：
    1. 按键(Pin53)选取目标位置：按下瞬间捕获当前小球位置作为目标位置
    2. 串口协议扩展：
       - 实时帧 0x2C 0x5A data 0xFE (沿用V1.3)
       - 目标帧 0x2C 0x5B data 0xFE (新增)
    3. 按键后连发5帧目标帧，之后切回实时帧
    4. 无目标时 data=0x00；按下时无球则清空目标并连发5帧0x00
    5. 屏幕上用纯蓝十字标记捕获的目标位置(持续显示至下次按下或清空)
    6. LED反馈：捕获成功绿灯闪，无球按下红灯闪

继承V1.3：检测去重(本地NMS)、速度预测跟踪、Coasting滑行、邻近抑制、单目标硬限制
'''

import os, gc
import time
import ustruct
from machine import UART
from machine import FPIOA
from machine import Pin
from libs.PlatTasks import DetectionApp
from libs.PipeLine import PipeLine
from libs.Utils import *

# ===================== 串口初始化 =====================
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.UART2_TXD)
fpioa.set_function(12, FPIOA.UART2_RXD)
uart = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# ===================== GPIO：按键 + LED =====================
fpioa.set_function(53, FPIOA.GPIO53)   # 按键
fpioa.set_function(20, FPIOA.GPIO20)   # 绿灯
fpioa.set_function(62, FPIOA.GPIO62)   # 红灯
fpioa.set_function(63, FPIOA.GPIO63)   # 蓝灯

# LED 共阳：高电平熄灭，低电平亮
LED_R = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive=7)  # 红灯
LED_G = Pin(20, Pin.OUT, pull=Pin.PULL_NONE, drive=7)  # 绿灯
LED_B = Pin(63, Pin.OUT, pull=Pin.PULL_NONE, drive=7)  # 蓝灯
LED_R.high()
LED_G.high()
LED_B.high()

# 按键：按下为高电平
button = Pin(53, Pin.IN, Pin.PULL_DOWN)
debounce_delay = 200       # 按键消抖时长(ms)
last_press_time = 0
button_last_state = 0
# ========================================================

# ===================== 串口发送函数 =====================
def _encode_dist_byte(dist_cm):
    """cm距离 -> 补码字节(dist*10, 限制在-128~127)"""
    dist_x10 = int(round(dist_cm * 10))
    if dist_x10 > 127:
        dist_x10 = 127
    elif dist_x10 < -128:
        dist_x10 = -128
    return dist_x10 & 0xFF

def send_distance(dist_cm):
    """发送实时距离帧: 0x2C 0x5A data 0xFE"""
    data_byte = _encode_dist_byte(dist_cm)
    uart.write(ustruct.pack(">BBBB", 0x2C, 0x5A, data_byte, 0xFE))

def send_target(dist_cm):
    """发送目标距离帧: 0x2C 0x5B data 0xFE (dist_cm=0.0 -> data=0x00 表示无目标)"""
    data_byte = _encode_dist_byte(dist_cm)
    uart.write(ustruct.pack(">BBBB", 0x2C, 0x5B, data_byte, 0xFE))
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

# ---------- 目标位置捕获状态 ----------
TARGET_SEND_N = 5          # 按键后目标帧连发帧数
target_dist_cm = 0.0       # 捕获的目标距离(cm)
has_target = False         # 是否已捕获目标
target_send_frames = 0     # 目标帧剩余发送次数
target_cx_src = 0          # 目标位置源坐标x(用于画十字)
target_cy_src = 0          # 目标位置源坐标y

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

        # ---------- 按键检测（上升沿 + 消抖）→ 捕获/清空目标 ----------
        current_time = time.ticks_ms()
        button_state = button.value()
        if button_state == 1 and button_last_state == 0:
            if current_time - last_press_time > debounce_delay:
                last_press_time = current_time
                if len(stable) > 0:
                    # 有球：捕获当前球位置作为目标
                    target_dist_cm = pixel_to_cm(int(stable[0].cx))
                    target_cx_src = int(stable[0].cx)
                    target_cy_src = int(stable[0].cy)
                    has_target = True
                    target_send_frames = TARGET_SEND_N
                    LED_G.low()
                    time.sleep_ms(20)
                    LED_G.high()
                    print(f"[目标] 已捕获 dist={target_dist_cm}cm 位置=({target_cx_src},{target_cy_src})")
                else:
                    # 无球：清空目标
                    has_target = False
                    target_send_frames = TARGET_SEND_N
                    LED_R.low()
                    time.sleep_ms(20)
                    LED_R.high()
                    print("[目标] 无球，目标已清空")
        button_last_state = button_state

        # ---------- 串口发送（目标帧优先连发N帧，之后切实时帧）----------
        if target_send_frames > 0:
            # 目标帧：有目标发捕获距离，无目标发0.0(data=0x00)
            send_target(target_dist_cm if has_target else 0.0)
            target_send_frames -= 1
        else:
            # 实时帧：有球发距离，没球不发（沿用V1.3）
            if len(stable) > 0:
                send_distance(pixel_to_cm(int(stable[0].cx)))

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

        # 绘制目标位置十字（纯蓝，仅当已捕获目标时持续显示）
        if has_target:
            try:
                tcx = int(target_cx_src * scale_x)
                tcy = int(target_cy_src * scale_y)
                pl.osd_img.draw_cross(tcx, tcy, (0, 0, 255), 12, 3)
            except:
                pass

        # 绘制目标位置文字（右下角，竖排逐字绘制，带符号和单位，红色）
        try:
            target_str = f"目标:{target_dist_cm:+.1f}cm" if has_target else "目标:无"
            _char_size = 28
            _step = _char_size + 4                              # 行距，留少量间隙避免重叠
            _block_h = len(target_str) * _step
            _ty = dst_h - _block_h - 8                          # 底部留8px，自下向上排
            _tx = dst_w - _char_size - 8                        # 右侧留8px
            for i, ch in enumerate(target_str):
                pl.osd_img.draw_string_advanced(_tx, _ty + i * _step, _char_size, ch, color=(255, 0, 0))
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
