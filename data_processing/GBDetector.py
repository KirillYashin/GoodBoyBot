import cv2
import numpy as np
from openvino.inference_engine import IECore
import sys
import logging as log
import argparse
import pathlib
from time import time

from GBClassifier import prepairing_classification_model
from GBClassifier import dog_classifier


#from performance_metrics import PerformanceMetrics

sys.path.append('C:\\Program Files (x86)\\Intel\\openvino_2021\\deployment_tools\\open_model_zoo\\demos\\common\\python')
import models
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
    
    # ДУБЛИРУЮЩИЕ ФУНКЦИИ
    '''def detection():
        # Load YOLOv3 model
        detector = models.YOLO(ie, pathlib.Path(args.model), None, 
                                threshold=args.prob_threshold, keep_aspect_ratio=True)
        
        # Initialize async pipeline
        detector_pipeline = AsyncPipeline(ie, detector, plugin_configs, device='CPU', 
                                            max_num_requests=1)'''

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

# Detection for pytorch - Unused
'''def pet_detection(out, frame, threshold):
    (height, width) = frame.shape[:2]
    cats = []
    dogs = []
    for detection in out.reshape(-1, 7):
        det_class = int(detection[1])
        confidence = float(detection[2])
        if det_class == 15 or det_class == 16:
            if confidence > threshold:
                xmin = max(int(detection[3] * width), 0)
                ymin = max(int(detection[4] * height), 0)
                xmax = min(int(detection[5] * width), width)
                ymax = min(int(detection[6] * height), height)
                if det_class == 16:
                    dogs.append(frame[ymin:ymax, xmin:xmax])
                    log.info("New dog has deteced")
                else:
                    cats.append(frame[ymin:ymax, xmin:xmax])
                    log.info("New cat has deteced")
    
    i = 0
    for dog in dogs:
        cv2.imshow('Dog #{}'.format(i), dog)
        i+=1
    i=0
    for cat in cats:
        cv2.imshow('Cat #{}'.format(i), cat)
        i+=1  
        
    return dogs, cats


def pytorch_detection(frame, netsize, net, thr):
    # Prepare input blob and perform an inference.
    blob = cv2.dnn.blobFromImage(frame, 1.0, netsize, ddepth = cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()

    return pet_detection(out, frame, thr)'''
 
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
                    check, result, cl_time = dog_classifier(ie, frame[ymin:ymax, xmin:xmax], dogs_count+1)
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

    # Showing Croped images
    '''i = 0
    for dog in dogs:
        cv2.imshow('Dog #{}'.format(i), dog)
        i+=1
    i=0
    for cat in cats:
        cv2.imshow('Cat #{}'.format(i), cat)
        i+=1'''
    
    return dogs, cats, (pr_time + classification_time)


def Detection(det_image):
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start OpenVINO object detection")
    start = time()
    # Initialize data input
    #cap = open_images_capture(args.input, True)
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
        if cv2.waitKey(0) & 0xFF == 27:
            break
        cv2.destroyAllWindows()
        
        # Get one image 
        #img = cap.read()
        while True:
            #img_path = input("Enter a path of your image ")
            # Нужно будет включить автоматическую подгрузку и выгрузку через бот
            try:
                #img = cv2.imread(img_path)
                img = det_image

                cv2.imshow("out",img)
                cv2.destroyAllWindows()
                detection_start = time()
                frame_id = 0
                detector_pipeline.submit_data(img,frame_id,{'frame':img,'start_time':0})
                detector_pipeline.await_any()
                break
            except Exception as ex:
                print(img_path, " can not be open. Try another path, please ")
                continue
        
        # Get detection result
        results, meta = detector_pipeline.get_result(frame_id)

        # Get list of detections in the image
        if results:
            Dogs, Cats, classification_time = yolo_detection(img, results, args.prob_threshold)
            detection_end = time()
        
            cv2.imshow('Detections', img)
            #result = list_classifier(Dogs)
        else:
            detection_end = time()
        end = time()

        # Time usage
        log.info("Classification time: {} sec".format(int(classification_time * 100)/100))
        log.info("Detection with classification checking: {} sec".format(int((detection_end - detection_start)*100)/100))
        log.info("Usage time: {} sec".format(int((end - start)*100)/100))
        print("Type any button to continue usage the God Boy Bot or type ESC to exit")
        
    # Destroy all windows
    cv2.destroyAllWindows()
    return

def main():
    Detection(image)

if __name__ == '__main__':
    sys.exit(main())
