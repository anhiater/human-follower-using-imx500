import argparse
import multiprocessing
import queue
import threading
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO

from functools import lru_cache
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# FPS Tracking Variables
prev_time = time.time()
fps = 0

# GPIO Pin Definitions
TRIG, ECHO = 23, 24

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    """Measure distance using the HC-SR04 ultrasonic sensor."""
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    start_time, stop_time = time.time(), time.time()
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    
    return max(2, min(400, ((stop_time - start_time) * 34300) / 2))

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return None
    
    threshold, max_detections = args.threshold, args.max_detections
    boxes, scores, classes = postprocess_nanodet_detection(np_outputs[0], conf=threshold, iou_thres=args.iou, max_out_dets=max_detections)[0]
    from picamera2.devices.imx500.postprocess import scale_boxes
    boxes = scale_boxes(boxes, 1, 1, *imx500.get_input_size(), False, False)
    
    detections = [Detection(box, category, score, metadata) for box, score, category in zip(boxes, scores, classes) if score > threshold and category == 0]
    return [max(detections, key=lambda d: d.conf)] if detections else None

@lru_cache
def get_labels():
    return [label for label in intrinsics.labels if label and label != "-"] if intrinsics.ignore_dash_labels else intrinsics.labels

def draw_detections(jobs):
    global prev_time, fps
    labels = get_labels()
    last_detections = []
    
    while True:
        job = jobs.get()
        if job is None:
            break
        
        request, async_result = job
        detections = async_result.get() or last_detections
        last_detections = detections
        distance = get_distance()

        with MappedArray(request, 'main') as m:
            h, w, _ = m.array.shape
            frame_center = (w // 2, h // 2)
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            
            cv2.putText(m.array, f"FPS: {fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(m.array, frame_center, 5, (255, 0, 0), -1)
            
            for detection in detections:
                x, y, w, h = detection.box
                obj_center = (x + w // 2, y + h // 2)
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(m.array, obj_center, 5, (0, 0, 255), -1)
                cv2.line(m.array, frame_center, obj_center, (0, 255, 255), 2)
                cv2.putText(m.array, f"{labels[detection.category]} ({detection.conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            bar_x, bar_h = w - 50, int((distance / 200) * 200)
            cv2.rectangle(m.array, (bar_x, 50), (bar_x + 20, 250), (50, 50, 50), -1)
            cv2.rectangle(m.array, (bar_x, 250 - min(200, max(10, bar_h))), (bar_x + 20, 250), (0, 255, 0), -1)
            cv2.putText(m.array, f"{distance:.1f} cm", (bar_x - 40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow('IMX500 Person Detection', m.array)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        request.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = argparse.ArgumentParser().parse_args()
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    picam2 = Picamera2(imx500.camera_num)
    picam2.start(picam2.create_preview_configuration({'format': 'RGB888'}, buffer_count=12), show_preview=False)
    
    pool = multiprocessing.Pool(processes=4)
    jobs = queue.Queue()
    threading.Thread(target=draw_detections, args=(jobs,)).start()
    
    try:
        while True:
            request = picam2.capture_request()
            metadata = request.get_metadata()
            if metadata:
                jobs.put((request, pool.apply_async(parse_detections, (metadata,))))
            else:
                request.release()
    except KeyboardInterrupt:
        GPIO.cleanup()
        sys.exit()


#packages to install :
#sudo apt-get install imx500-all
#sudo update -y
#pip install opencv-python numpy RPi.GPIO
#pip install gpiozero

