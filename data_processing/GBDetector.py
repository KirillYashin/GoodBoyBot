import cv2
import sys
import json
import pathlib
import argparse
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IECore

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
    

counter = 0


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
    dogs_pics = []
    dogs_count = 0
    cats_count = 0
    classification_time = 0
    conf = 0
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
                    check, breeds, conf, cl_time = dog_classifier(ie, frame[ymin:ymax, xmin:xmax], dogs_count+1)
                    classification_time += cl_time
                    if check:
                        dogs_count += 1
                        dogs_pics.append(frame[ymin:ymax, xmin:xmax])
                        dogs.append(breeds)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.rectangle(frame, (xmin, ymin+30), (xmax, ymin), (0, 255, 0), -1)
                        cv2.putText(frame, translator(str(breeds[0])),(xmin, ymin+23), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.rectangle(frame, (xmin, ymax), (xmax, ymax-30), (0, 255, 0), -1)
                        cv2.putText(frame, f"Уверенность: {round(conf[0]*100, 1)}%",(xmin, ymax-7), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    cats_count += 1
                    cats.append(frame[ymin:ymax, xmin:xmax])
                    '''cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.rectangle(frame, (xmin, ymin-30), (xmin+95, ymin), (0, 0, 255), -1)
                    cv2.putText(frame, ' Cat #{}'.format(cats_count),(xmin, ymin - 7), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)'''
    
    return dogs, cats, conf, (pr_time + classification_time)


def translator(en_breed):
    with open("..\\data\\breeds_ru_to_en.json", "r", encoding="utf-8") as translator:
        breeds = json.load(translator)
        for ru_breed, val in breeds.items():
            if val == str(en_breed):
                return ru_breed


def get_breed_info(ru_breed):
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    global counter
    counter += 1
    log.info(f"The function was used {counter} times")
    with open("..\\data\\breeds_ru_to_en.json", "r", encoding="utf-8") as translator:
        en_breeds = json.load(translator)
        en_breed = en_breeds.get(str(ru_breed))
        with open("..\\data\\data.json", "r", encoding="utf-8") as read_file:
            data = json.load(read_file)
            breed_info = data.get(en_breed, None)
            if not breed_info:
                info = " "
                link = " "
            else:
                link = breed_info.get("Ссылка на картинку")
                info = breed_info.get("Описание") + '\n' + \
                        "Стоимость: " + breed_info.get("Стоимость") + '\n' + \
                        "Длительность жизни: " + breed_info.get("Длительность жизни") + '\n' + \
                        "Шерсть: " + breed_info.get("Шерсть") + '\n' + \
                        "Рождаемость: " + breed_info.get("Рождаемость") + '\n' + \
                        "Рост в холке, см: " + breed_info.get("Рост в холке, см") + '\n' + \
                        "Вес, кг: " + breed_info.get("Вес, кг") + '\n' + \
                        "Опыт содержания: " + breed_info.get("Содержание") + '\n' + \
                        "Назначение: " + breed_info.get("Назначение") + '\n' + \
                        "Ссылка на собаку: " + breed_info.get("Ссылка на собаку") + '\n'
            return info, link


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

    img = det_image

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
                
        # cv2.imshow('The outcome', outcome_image)
        # cv2.waitKey(0)
    else:
        log.info("I see nothing")
        Confidences = 0
        classification_time = 0
        outcome_image=img
        detection_end = time()
    end = time()

    # Time usage
    log.info("Classification time: {} sec".format(round(classification_time, 2)))
    log.info("Detection with classification checking: {} sec".format(round(detection_end - detection_start), 2))
    log.info("Usage time: {} sec".format(round((end - start), 2)))

    # Destroy all windows
    cv2.destroyAllWindows()
    return Dogs, Cats, Confidences, outcome_image


def main():
    Detection(image)


if __name__ == '__main__':
    sys.exit(main())
