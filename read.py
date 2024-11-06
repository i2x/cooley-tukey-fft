import serial
import struct

# ตั้งค่าการเชื่อมต่อพอร์ตอนุกรมสำหรับ COM3
ser = serial.Serial('COM3', baudrate=115200, timeout=1)

# อ่านข้อมูลทีละ 2 ไบต์จาก UART และแปลงเป็น 16 บิต
try:
    while True:
        # อ่าน 2 ไบต์ (16 บิต) จากพอร์ตอนุกรม
        raw_data = ser.read(2)
        if len(raw_data) == 2:
            # แปลงข้อมูลเป็น 16 บิต (Big-endian)
            value = struct.unpack('>H', raw_data)[0]
            print(f"16-bit value: {value:#06x}")
        else:
            print("ไม่พบข้อมูลเพียงพอ")
            break
finally:
    ser.close()  # ปิดพอร์ตอนุกรมเมื่อเสร็จสิ้น