import time
from machine import UART
from machine import FPIOA
import ustruct

# 引脚配置
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.UART2_TXD)
fpioa.set_function(12, FPIOA.UART2_RXD)

# 串口初始化
uart = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

def send_uart(data_val):
    """
    帧结构：0x2C(帧头1) + 0x5A(帧头2) + 单字节数据 + 0xFE(帧尾)
    格式：>BBBB
    BB：2字节帧头
    B：1字节有效数据
    B：1字节帧尾
    """
    frame = ustruct.pack(">BBBB",
                         0x2C, 0x5A,    # 帧头
                         int(data_val), # 唯一发送变量
                         0xFE)          # 帧尾
    uart.write(frame)
    return frame

# 主循环
while True:
    send_data = 1   # 需要发送的变量，直接在这里赋值
    send_frame = send_uart(send_data)
    # print("发送帧：", send_frame)
    print("发送帧十六进制：", " ".join("0x{:02X}".format(b) for b in send_frame))
    time.sleep(1)

# uart.deinit()
