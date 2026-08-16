#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MaixCAM2 steel-ball detection + UART report  (YOLO26, single class 'ball')

Copy onto the board:
    /root/models/ball_v5_1024.mud
    /root/models/ball_v5_1024_npu.axmodel
    /root/models/ball_v5_1024_vnpu.axmodel

Tuning notes at the bottom of the file.
"""

import struct

from maix import app, camera, display, image, nn, uart, pinmap, err, time

# ----------------------------------------------------------------- config
MODEL = "/root/models/ball_v5_1024.mud"

# Leave False until the MSPM0 is wired up, or the missing UART raises on boot.
ENABLE_UART = False

# Small/far balls detect in only ~half the frames (FLICKER), but confidence
# when they DO hit is healthy. Keep the floor low so a dim small ball still
# passes on its hit frames; the tracker below rejects lone strays by requiring
# a couple of consistent hits before it draws green.
CONF_TH = 0.20

# yolo26 uses a one2one (NMS-free) head, so this IoU is retained only for API
# compatibility and has little effect. Kept at 0.7 to match the v2 script.
IOU_TH = 0.7

# ---- tracker (fixes the "green box flickers / vanishes on small balls") ----
# ROOT CAUSE of the old blink: the previous voter only emitted a box when the
# CURRENT frame had a detection. On any miss frame -> no green box that frame ->
# the box blinks off. A small ball misses ~every other frame, so it strobes.
#
# The tracker keeps a CONFIRMED target on screen through miss frames by COASTING
# it at its last known position for up to COAST_MAX frames. A 50%-duty ball
# never stacks COAST_MAX consecutive misses, so its green box stays lit
# continuously instead of strobing. EMA smoothing kills the per-frame jitter.
CONFIRM_HITS = 2     # hits before a track is drawn (rejects 1-frame strays)
COAST_MAX = 8        # keep drawing a confirmed track through up to N miss frames (~0.4s @20fps)
MATCH_DIST = 55      # px; center distance that still counts as "the same ball"
EMA_ALPHA = 0.5      # position smoothing: new = a*meas + (1-a)*old

UART_DEV = "/dev/ttyS4"
UART_BAUD = 9600

# ----------------------------------------------------------------- uart
ser = None
if ENABLE_UART:
    err.check_raise(pinmap.set_pin_function("A21", "UART4_TX"), "set A21 failed")
    err.check_raise(pinmap.set_pin_function("A22", "UART4_RX"), "set A22 failed")
    ser = uart.UART(UART_DEV, UART_BAUD)


def calc_bcc(data):
    b = 0
    for x in data:
        b ^= x
    return b


def send_target(cam_w, cam_h, tx, ty):
    """K210 protocol, 11-byte frame: 0xCC + W + H + X + Y + BCC + 0xDD"""
    if ser is None:
        return
    payload = struct.pack("<BHHHH", 0xCC, cam_w, cam_h, int(tx), int(ty))
    ser.write(payload + bytes([calc_bcc(payload), 0xDD]))


# ----------------------------------------------------------------- tracking
class Track:
    __slots__ = ("cx", "cy", "w", "h", "score", "hits", "misses", "confirmed")

    def __init__(self, cx, cy, w, h, score):
        self.cx, self.cy, self.w, self.h, self.score = cx, cy, w, h, score
        self.hits, self.misses, self.confirmed = 1, 0, False


class Tracker:
    """Multi-target tracker with coasting.

    Confirmed targets keep being reported even on frames where the model missed
    them (up to COAST_MAX consecutive misses), drawn at their last known/EMA
    position. This is what stops the green box from strobing on flickery small
    balls. Returns the list of confirmed, still-alive tracks each frame.
    """

    def __init__(self):
        self.tracks = []

    def update(self, dets):
        used = [False] * len(dets)
        d2max = MATCH_DIST * MATCH_DIST
        # 1) match existing tracks to nearest unused detection
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
                t.misses += 1  # coasting: hold last position, count the miss
        # 2) spawn tracks for detections that matched nothing
        for i, d in enumerate(dets):
            if not used[i]:
                self.tracks.append(Track(*d))
        # 3) drop tracks that have coasted too long (target truly gone)
        self.tracks = [t for t in self.tracks if t.misses <= COAST_MAX]
        # 4) report confirmed survivors (including those currently coasting)
        return [t for t in self.tracks if t.confirmed]


# ----------------------------------------------------------------- main
# dual_buff=False was the config proven on v2. Try True later for more fps,
# but treat it as an unverified change -- revert to False if anything breaks.
detector = nn.YOLO26(model=MODEL, dual_buff=False)
cam = camera.Camera(detector.input_width(), detector.input_height(),
                    detector.input_format())
disp = display.Display()
tracker = Tracker()

W, H = detector.input_width(), detector.input_height()
print(f"[ball] model loaded, input {W}x{H}, labels={detector.labels}")

fps_t = time.ticks_ms()
frames = 0

while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)

    dets = [(o.x + o.w // 2, o.y + o.h // 2, o.w, o.h, o.score) for o in objs]
    stable = tracker.update(dets)

    # faint blue = this frame's raw detections (will flicker -- that's expected)
    for o in objs:
        img.draw_rect(o.x, o.y, o.w, o.h, image.COLOR_BLUE, 1)

    # green = confirmed tracks. Drawn every frame, INCLUDING coast frames where
    # the model missed the ball, so the box stays lit instead of strobing.
    for t in stable:
        x = int(t.cx - t.w / 2)
        y = int(t.cy - t.h / 2)
        w = int(t.w)
        h = int(t.h)
        img.draw_rect(x, y, w, h, image.COLOR_GREEN, 2)
        img.draw_string(x, max(0, y - 14), f"{t.score:.2f}", image.COLOR_GREEN)

    if stable:
        # Report the largest (nearest/clearest). Change this line if the task
        # wants a different pick (most central, leftmost, ...).
        tgt = max(stable, key=lambda t: t.w * t.h)
        send_target(W, H, int(tgt.cx), int(tgt.cy))
        img.draw_cross(int(tgt.cx), int(tgt.cy), image.COLOR_RED, 8, 2)

    img.draw_string(4, 4, f"{len(objs)} det / {len(stable)} tracked",
                    image.COLOR_RED)
    disp.show(img)

    frames += 1
    if frames % 30 == 0:
        now = time.ticks_ms()
        print(f"[ball] {30000.0 / max(1, now - fps_t):.1f} fps")
        fps_t = now
