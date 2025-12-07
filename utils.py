import serial
import time

PORT = '/dev/ttyUSB0'  # Change this to your port (e.g., 'COM3' for Windows)
BAUDRATE = 115200

def read_serial(ser, duration=2):
    """Read from serial for a specified duration"""
    start_time = time.time()
    while time.time() - start_time < duration:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='ignore')
            if data:
                print(f"Received: {data}", end='')
        time.sleep(0.01)  # Small delay to prevent CPU spinning

def call_serial(command: str):
    ser = serial.Serial(PORT, baudrate=BAUDRATE, dsrdtr=None)
    ser.setRTS(False)
    ser.setDTR(False)
    
    try:
        # Send the command once
        ser.write(command.encode() + b'\n')
        print(f"Sent: {command}")
        
        # Wait for and read response
        read_serial(ser, duration=2)  # Wait up to 2 seconds for response
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        ser.close()
        print("Serial port closed")