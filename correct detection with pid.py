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
        """Create a Detection object, recording the bounding box, category, and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    bbox_normalization = intrinsics.bbox_normalization
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return None
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    # Filter detections to include only "person" category (assuming label 0 is for 'person')
    detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold and category == 0  # Ensure only 'person' is detected
    ]

    # Keep only the most confident detection if multiple people detected
    if detections:
        detections = [max(detections, key=lambda d: d.conf)]

    return detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(jobs):
    """Draw the detections for this request onto the ISP output."""
    global prev_time, fps  # For FPS calculation
    labels = get_labels()
    last_detections = []

    while True:
        job = jobs.get()
        if job is None:
            break  # Exit the loop if None is received (graceful shutdown)

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
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
                yaw_error = frame_center[0] - object_center[0]
                pitch_error = frame_center[1] - object_center[1]
                # Draw circle at object center
                cv2.circle(m.array, object_center, 5, (0, 0, 255), -1)  # Red dot in detected person
                
                # Draw label
                cv2.putText(m.array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw a line from the frame center to detected object
                cv2.line(m.array, frame_center, object_center, (0, 255, 255), 2)  # Yellow line

                distance = sensor.distance
                if distance is None:
                    continue
                distance_error = 1.5 - distance
                cv2.putText(m.array, distance_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                send_mavlink_velocity(yaw_pid(yaw_error), pitch_pid(pitch_error), throttle_pid(distance_error))
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Tracking', m.array)
                if cv2.waitKey(1) & 0xFF == 27:
                    land()
        request.release()
    cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set IoU threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print network intrinsics and exit")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = get_args()

        # Initialize AI camera and set up parameters
        imx500 = IMX500(args.model)
        intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"

        # Load labels if specified
        if args.labels:
            with open(args.labels, "r") as f:
                intrinsics.labels = f.read().splitlines()

        if args.print_intrinsics:
            print(intrinsics)
            exit()

        # Set up camera
        picam2 = Picamera2(imx500.camera_num)
        main = {'format': 'RGB888'}
        config = picam2.create_preview_configuration(main, controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

        picam2.start(config, show_preview=False)

        pool = multiprocessing.Pool(processes=4)
        jobs = queue.Queue()

        thread = threading.Thread(target=draw_detections, args=(jobs,))
        thread.start()
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
GPIO.cleanup()
