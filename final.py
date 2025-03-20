import argparse
import multiprocessing
import queue
import sys
import threading
import time
from functools import lru_cache
import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# FPS Tracking Variables
prev_time = time.time()
fps = 0


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

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS on the frame
            fps_text = f"FPS: {fps:.2f}"
            cv2.rectangle(m.array, (10, 10), (120, 40), (0, 0, 0), -1)  # Background rectangle
            cv2.putText(m.array, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw center of frame
            cv2.circle(m.array, frame_center, 5, (255, 0, 0), -1)  # Blue dot in center

            for detection in detections:
                x, y, w, h = detection.box
                object_center = (x + w // 2, y + h // 2)

                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                # Draw bounding box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                # Draw circle at object center
                cv2.circle(m.array, object_center, 5, (0, 0, 255), -1)  # Red dot in detected person

                # Draw a line from the frame center to detected object
                cv2.line(m.array, frame_center, object_center, (0, 255, 255), 2)  # Yellow line

                # Draw label
                cv2.putText(m.array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('IMX500 Person Detection', m.array)

            # Exit when 'Esc' is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break  # Break the loop to stop the program

        request.release()

    cv2.destroyAllWindows()  # Close OpenCV window


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

    while True:
        request = picam2.capture_request()
        metadata = request.get_metadata()

        if metadata:
            async_result = pool.apply_async(parse_detections, (metadata,))
            jobs.put((request, async_result))
        else:
            request.release()
