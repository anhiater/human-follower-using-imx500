import argparse
import multiprocessing
import queue
import sys
import threading
import time
from functools import lru_cache
import cv2
import numpy as np
import subprocess
import serial
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
from time import sleep
sensor = DistanceSensor(echo=24, trigger=23, max_distance=4)
from pymavlink import mavutil
from simple_pid import PID

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Initialize serial connection to Pixhawk
try:
    pixhawk = mavutil.mavlink_connection('/dev/serial0', baud=115200)
    print("Waiting for heartbeat from Pixhawk...")
    pixhawk.wait_heartbeat(timeout=5)
    print("Heartbeat received, connection established.")
except Exception as e:
    print(f"Failed to connect to Pixhawk: {e}")
    sys.exit(1)

# Initialize PID controllers for yaw, pitch, and throttle
yaw_pid = PID(0.01, 0.001, 0.0005, setpoint=0)
pitch_pid = PID(0.02, 0.001, 0.0005, setpoint=0)
throttle_pid = PID(0.05, 0.002, 0.001, setpoint=1.5)  # Maintain 1.5m distance
yaw_pid.output_limits = (-30, 30)
pitch_pid.output_limits = (-10, 10)
throttle_pid.output_limits = (-0.5, 0.5)

TARGET_ALTITUDE = 2.0  # Meters

# Initialize ultrasonic sensor
TRIG_PIN = 23
ECHO_PIN = 24
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def get_distance():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    start_time = time.time()
    stop_time = time.time()
    timeout = start_time + 0.1  # 100ms timeout
    
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
        if start_time > timeout:
            return None

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
        if stop_time > timeout:
            return None
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Convert to cm
    distance_m = distance / 100
    
    if distance_m < 0.2 or distance_m > 3.0:
        return None
    return distance_m

# Function to send velocity commands to Pixhawk
def send_mavlink_velocity(yaw_rate, pitch_rate, throttle_adj):
    msg = pixhawk.mav.set_attitude_target_encode(
        0, 1, 1, 0b00000100, [1, 0, 0, 0], 0, pitch_rate, yaw_rate, throttle_adj
    )
    pixhawk.send_mavlink(msg)
    pixhawk.flush()

# Function to take off
def arm_and_takeoff(target_altitude):
    pixhawk.arducopter_arm()
    pixhawk.motors_armed_wait()
    print("Armed, taking off...")
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, target_altitude
    )
    time.sleep(10)

def land():
    print("Landing...")
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(10)
    GPIO.cleanup()
    sys.exit(0)

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return None
    boxes, scores, classes = postprocess_nanodet_detection(np_outputs[0], conf=args.threshold, iou_thres=args.iou, max_out_dets=args.max_detections)[0]
    boxes = np.array(boxes)
    detections = [Detection(box, category, score, metadata) for box, score, category in zip(boxes, scores, classes) if score > args.threshold and category == 0]
    return [max(detections, key=lambda d: d.conf)] if detections else []

def draw_detections(jobs):
    labels = intrinsics.labels
    last_detections = []
    while True:
        job = jobs.get()
        if job is None:
            break
        request, async_result = job
        detections = async_result.get()
        if detections is None:
            detections = last_detections
        last_detections = detections
        with MappedArray(request, 'main') as m:
            frame_center = (m.array.shape[1] // 2, m.array.shape[0] // 2)
            for detection in detections:
                x, y, w, h = detection.box
                object_center = (x + w // 2, y + h // 2)
                yaw_error = frame_center[0] - object_center[0]
                pitch_error = frame_center[1] - object_center[1]
                distance = get_distance()
                if distance is None:
                    continue
                distance_error = 1.5 - distance
                send_mavlink_velocity(yaw_pid(yaw_error), pitch_pid(pitch_error), throttle_pid(distance_error))
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Tracking', m.array)
                if cv2.waitKey(1) & 0xFF == 27:
                    land()
        request.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        arm_and_takeoff(TARGET_ALTITUDE)
        while True:
            request = picam2.capture_request()
            metadata = request.get_metadata()
            if metadata:
                async_result = pool.apply_async(parse_detections, (metadata,))
                jobs.put((request, async_result))
            else:
                request.release()
    except KeyboardInterrupt:
        land()
