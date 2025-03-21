import argparse
import multiprocessing
import queue
import sys
import threading
import time
import cv2
import numpy as np
import serial
from pymavlink import mavutil
from simple_pid import PID

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Initialize serial connection to Pixhawk
pixhawk = mavutil.mavlink_connection('/dev/serial0', baud=115200)
time.sleep(2)  # Allow connection to establish

# Initialize PID controllers for yaw, pitch, and throttle
yaw_pid = PID(0.01, 0.001, 0.0005, setpoint=0)
pitch_pid = PID(0.02, 0.001, 0.0005, setpoint=0)
throttle_pid = PID(0.05, 0.002, 0.001, setpoint=1.5)  # Maintain 1.5m distance
yaw_pid.output_limits = (-30, 30)  # Limit yaw correction
pitch_pid.output_limits = (-10, 10)  # Limit pitch correction
throttle_pid.output_limits = (-0.5, 0.5)  # Limit throttle adjustment

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
    
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2  # Convert to cm
    return distance / 100  # Convert to meters

# Function to send velocity commands to Pixhawk
def send_mavlink_velocity(yaw_rate, pitch_rate, throttle_adj):
    msg = pixhawk.mav.set_attitude_target_encode(
        0,  # Time boot ms
        1,  # Target system
        1,  # Target component
        0b00000100,  # Type mask (ignore roll, control yaw)
        [1, 0, 0, 0],  # Quaternion (not used)
        0,  # Roll rate (not controlled)
        pitch_rate,  # Pitch rate control
        yaw_rate,  # Yaw rate control
        throttle_adj  # Thrust adjustment
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
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, target_altitude
    )
    time.sleep(10)  # Wait for takeoff to stabilize

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
                distance_error = 1.5 - distance
                yaw_correction = yaw_pid(yaw_error)
                pitch_correction = pitch_pid(pitch_error)
                throttle_correction = throttle_pid(distance_error)
                send_mavlink_velocity(yaw_correction, pitch_correction, throttle_correction)
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(m.array, object_center, 5, (0, 0, 255), -1)
                cv2.putText(m.array, f"Yaw: {yaw_correction:.2f}, Pitch: {pitch_correction:.2f}, Dist: {distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Tracking', m.array)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        request.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser().parse_args()
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration({'format': 'RGB888'}, buffer_count=12)
    picam2.start(config, show_preview=False)
    arm_and_takeoff(TARGET_ALTITUDE)
    pool = multiprocessing.Pool(processes=4)
    jobs = queue.Queue()
    thread = threading.Thread(target=draw_detections, args=(jobs,))
    thread.start()
    while True:
        request = picam2.capture_request()
        metadata = request.get_metadata()
        if metadata:
            async_result = pool.apply_async(parse_detections, (metadata,))
            jobs.put((request, async_result))
        else:
            request.release()
    GPIO.cleanup()
