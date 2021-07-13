
import cv2
import numpy as np
from openvino.inference_engine import IECore
import sys
import logging as log
import argparse
import pathlib
import random
import colorsys
#from performance_metrics import PerformanceMetrics

sys.path.append('C:\\Program Files (x86)\\Intel\\openvino_2021\\deployment_tools\\open_model_zoo\\demos\\common\\python')
import models
from pipelines import AsyncPipeline
from images_capture import open_images_capture

class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


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
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-o', '--output', required=False, default='./saved/',
                        help='Optional. Name of the output file(s) to save.')    
    parser.add_argument('-t', '--prob_threshold', default=0.5, type=float,
        help='Optional. Probability threshold for detections filtering.')
    return parser
  
  
def crop_pets(frame, detections, threshold):
    size = frame.shape[:2]
    cats = []
    dogs = []
    for detection in detections:
        det_class = int(detection.id)
        # If score more than threshold and detection is a pet
        if det_class == 15 or det_class == 16:
            if detection.score > threshold:
                xmin = max(int(detection.xmin), 0)
                ymin = max(int(detection.ymin), 0)
                xmax = min(int(detection.xmax), size[1])
                ymax = min(int(detection.ymax), size[0])
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
    
    '''cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
    cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)'''    
        
    return dogs, cats


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Start OpenVINO object detection")

    # Initialize data input
    cap = open_images_capture(args.input, True)
    # Initialize OpenVINO
    ie = IECore()
    # Initialize Plugin configs
    plugin_configs = get_plugin_configs('CPU', 0, 0)
    # Load YOLOv3 model
    detector = models.YOLO(ie, pathlib.Path(args.model), None, 
                            threshold=args.prob_threshold, keep_aspect_ratio=True)
    

    # Initialize async pipeline
    #metrics = PerformanceMetrics()
    detector_pipeline = AsyncPipeline(ie, detector, plugin_configs, device='CPU', 
                                        max_num_requests=1)
    Dogs = []
    Cats = []
    # Get one image 
    img = cap.read()
    # Start processing frame asynchronously
    frame_id = 0 
    detector_pipeline.submit_data(img,frame_id,{'frame':img,'start_time':0})
    detector_pipeline.await_any()
    # Get detection result
    results, meta = detector_pipeline.get_result(frame_id)

    # Get list of detections in the image
    if results:
        cv2.imshow('Original', img)
        Dogs, Cats = crop_pets(img, results, args.prob_threshold)

    cv2.waitKey(0)
      
    # Destroy all windows
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    sys.exit(main())
