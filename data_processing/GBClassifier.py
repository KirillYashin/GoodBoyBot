"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m resnet-50.xml -w resnet-50.bin -c imagenet_synset_words.txt
"""
import cv2
import sys
import json
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore


class InferenceEngineClassifier:
    def __init__(self, config_path=None, weights_path=None,
                 device='CPU', extension=None, classes_path=None):
        # Add code for Inference Engine initialization
        self.ie = IECore()
        # Add code for model loading
        self.net = self.ie.read_network(model=config_path)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        # Add code for classes names loading
        with open(classes_path, 'r') as f:
            self.labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        return

    def get_top(self, prob, top_n=1):
        result = np.squeeze(prob)
        result = np.argsort(result)[-top_n:][::-1]

        return result

    def _prepare_image(self, image, h, w):
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))

        return image

    def classify(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))

        n, c, h, w = self.net.inputs[input_blob].shape
        image = self._prepare_image(image, h, w)

        output = self.exec_net.infer(inputs={input_blob: image})

        output = output[out_blob]

        return output

def prepairing_classification_model():
    start = time()
    model = '..\\models\\public\\resnet-50-tf\\FP16\\resnet-50-tf.xml'
    weights = '..\\models\\public\\resnet-50-tf\\FP16\\resnet-50-tf.bin'
    classes_path = '..\\data\\imagenet_synset_words.txt'
    device = 'CPU'
    cpu_extension = 'CPU'

    ie_classifier = InferenceEngineClassifier(config_path=model, weights_path=weights,
                                              device=device, extension=cpu_extension,
                                              classes_path=classes_path)
    end = time()
    return ie_classifier, int((end-start)*100)/100

def dog_classifier(ie,researching_image, number):
    with open("..\\data\\data.json", "r", encoding="utf-8") as read_file:
        data = json.load(read_file)
    result = []
    breeds = []
    confidences = []
    start = time()
    prob = ie.classify(researching_image)
    end = time()
    predictions = ie.get_top(prob, 3)
    if (predictions[0] > 151 and predictions[0] < 269) or \
        ((predictions[1] > 151 and predictions[1] < 269) and prob[0][predictions[1]] > 0.15):
        check = True

        for i in range(3):
            if predictions[i] > 151 and predictions[i] < 269:
                breeds.append(ie.labels_map[predictions[i]-1])
                confidences.append(prob[0][predictions[i]])

        predictions = [str(ie.labels_map[predictions[i]-1]) + ': '
                        + str(predictions[i]) + "  with confidence "
                        + str(prob[0][predictions[i]]) for i in range(3)]
        result.append(predictions)
       
        #log.info("Predictions: " + str(predictions))
    else:
        check = False

    ranked = 0
    for prediction in result:
        print('Dog #{}:'.format(number))
        print("Top predictions:")
        for breed in result[ranked]:
            print('\t' + str(breed))
        ranked += 1
    #for i in range(result.shape):
    
    for breed in breeds:
        print(breed)
        breed_info = data.get(breed)
        for key, value in breed_info.items():
            print('\t',key, ': ', value)
    
    
    return check, breeds, confidences, int((end-start)*100)/100


def main():
    img1 = cv2.imread('img.png')
    img2 = cv2.imread('img_1.png')
    to_classify = [img1, img2]

    return


if __name__ == '__main__':
    sys.exit(main())
