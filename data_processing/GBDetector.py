import cv2
import numpy as np
from openvino.inference_engine import IECore
import sys
import logging as log
import argparse
import pathlib
from time import time

sys.path.append('C:\\Program Files (x86)\\Intel\\openvino_2021\\deployment_tools\\open_model_zoo\\demos\\common\\python')

import models

from GBClassifier import prepairing_classification_model
from GBClassifier import dog_classifier
from pipelines import AsyncPipeline
from images_capture import open_images_capture

class InferencEngineDetector:
    def __init__(self, device, num_streams, num_threads):

        # Initialize OpenVINO
        ie = IECore()

        self.config_user_specified = {}
        self.devices_nstreams = {}
        if num_streams:
            self.devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                if num_streams.isdigit() \
                else dict(device.split(':', 1) for device in num_streams.split(','))

        if 'CPU' in device:
            if num_threads is not None:
                self.config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
            if 'CPU' in self.devices_nstreams:
                self.config_user_specified['CPU_THROUGHPUT_STREAMS'] = self.devices_nstreams['CPU'] \
                    if int(self.devices_nstreams['CPU']) > 0 \
                    else 'CPU_THROUGHPUT_AUTO'

        if 'GPU' in device:
            if 'GPU' in self.devices_nstreams:
                self.config_user_specified['GPU_THROUGHPUT_STREAMS'] = self.devices_nstreams['GPU'] \
                    if int(self.devices_nstreams['GPU']) > 0 \
                    else 'GPU_THROUGHPUT_AUTO'

        return
    

def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-o', '--output', required=False, default='./saved/',
                        help='Optional. Name of the output file(s) to save.')    
    parser.add_argument('-t', '--prob_threshold', default=0.75, type=float,
        help='Optional. Probability threshold for detections filtering.')
    return parser

 
def yolo_detection(frame, detections, threshold):
    size = frame.shape[:2]
    cats = []
    dogs = []
    dogs_count = 0
    cats_count = 0
    classification_time = 0
    ie, pr_time = prepairing_classification_model()

    for detection in detections:
        det_class = int(detection.id)
        # If score more than threshold and detection is a pet
        if det_class == 15 or det_class == 16:
            if detection.score > threshold:
                xmin = max(int(detection.xmin), 0)
                ymin = max(int(detection.ymin), 0)
                xmax = min(int(detection.xmax), size[1])
                ymax = min(int(detection.ymax), size[0])
                if det_class == 16: # Dog
                    check, result, conf, cl_time = dog_classifier(ie, frame[ymin:ymax, xmin:xmax], dogs_count+1)
                    classification_time += cl_time
                    if check:
                        dogs_count += 1
                        dogs.append(frame[ymin:ymax, xmin:xmax])
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.rectangle(frame, (xmin, ymin-30), (xmin+95, ymin), (0, 255, 0), -1)
                        cv2.putText(frame, ' Dog #{}'.format(dogs_count),(xmin, ymin - 7), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    cats_count += 1
                    cats.append(frame[ymin:ymax, xmin:xmax])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.rectangle(frame, (xmin, ymin-30), (xmin+95, ymin), (0, 0, 255), -1)
                    cv2.putText(frame, ' Cat #{}'.format(cats_count),(xmin, ymin - 7), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
    
    return dogs, cats, conf, (pr_time + classification_time)


def Detector(det_image):
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start OpenVINO object detection")
    start = time()

    # Initialize data input
    Dogs = []
    Cats = []
    
    model_path = "..\\models\\public\\yolo-v3-tf\\FP16\\yolo-v3-tf.xml"
    # Initialize OpenVINO
    ie = IECore()
    # Initialize Plugin configs
    plugin_configs = get_plugin_configs('CPU', 0, 0)
    # Load YOLOv3 model
    detector = models.YOLO(ie, pathlib.Path(model_path), None, 
                            threshold=args.prob_threshold, keep_aspect_ratio=True)
        
    # Initialize async pipeline
    detector_pipeline = AsyncPipeline(ie, detector, plugin_configs, device='CPU', 
                                            max_num_requests=1)
    while True:

        # Get one image 
        #img_path = input("Enter a path of your image ")
        #img = cv2.imread(img_path)
        img = det_image

        #cv2.imshow("out",img)
        #cv2.destroyAllWindows()
        detection_start = time()
        frame_id = 0
        detector_pipeline.submit_data(img,frame_id,{'frame':img,'start_time':0})
        detector_pipeline.await_any()

        # Get detection result
        results, meta = detector_pipeline.get_result(frame_id)

        # Get list of detections in the image
        if results:
            Dogs, Cats, Confidences, classification_time = yolo_detection(img, results, args.prob_threshold)
            detection_end = time()

            outcome_image=img
            cv2.imshow('The outcome', outcome_image)
        else:
            detection_end = time()
        end = time()

        # Time usage
        log.info("Classification time: {} sec".format(int(classification_time * 100)/100))
        log.info("Detection with classification checking: {} sec".format(int((detection_end - detection_start)*100)/100))
        log.info("Usage time: {} sec".format(int((end - start)*100)/100))
        print("Type any button to continue usage the God Boy Bot or type ESC to exit")
        cv2.waitKey(0)
        break
    # Destroy all windows
    cv2.destroyAllWindows()
    return Dogs, Cats, Confidences, outcome_image

def main():
    Detection(image)

if __name__ == '__main__':
    sys.exit(main())
