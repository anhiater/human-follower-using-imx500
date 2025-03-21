import argparse
import multiprocessing
import queue
import sys
import threading
import time
import serial
from functools import lru_cache
import cv2
import numpy as np
from pymavlink import mavutil

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# PID Controller for Yaw Correction
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

# Initialize MAVLink connection
pixhawk = mavutil.mavlink_connection('/dev/serial0', baud=11520)
def send_yaw_command(yaw_rate):
    pixhawk.mav.manual_control_send(
        pixhawk.target_system, 0, 0, 500, int(yaw_rate * 1000), 0
    )

def arm_and_takeoff():
    pixhawk.wait_heartbeat()
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    time.sleep(3)
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 2
    )
    time.sleep(5)

# FPS Tracking Variables
prev_time = time.time()
fps = 0
pid_yaw = PID(0.02, 0.001, 0.01)  # Tune these values as needed

def draw_detections(jobs):
    global prev_time, fps
    labels = get_labels()
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
            frame_center_x = m.array.shape[1] // 2
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            if detections:
                detection = detections[0]
                x, _, w, _ = detection.box
                object_center_x = x + w // 2
                yaw_error = frame_center_x - object_center_x
                yaw_correction = pid_yaw.compute(yaw_error, 1 / fps)
                send_yaw_command(yaw_correction)

            cv2.imshow('IMX500 Person Detection', m.array)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        request.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    arm_and_takeoff()
    picam2 = Picamera2()
    picam2.start()
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
