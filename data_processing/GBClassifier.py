"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m resnet-50.xml -w resnet-50.bin -c imagenet_synset_words.txt
"""
import cv2
import sys
import numpy as np
import logging as log
from numpy.core.fromnumeric import rank
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
            self.labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip().split(sep=',')[0] for x in f]
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


def list_classifier(to_classify: list):
    model = '..\\models\\public\\resnet-50-tf\\FP16\\resnet-50-tf.xml'
    weights = '..\\models\\public\\resnet-50-tf\\FP16\\resnet-50-tf.bin'
    classes_path = '..\\data\\imagenet_synset_words.txt'
    device = 'CPU'
    cpu_extension = 'CPU'

    ie_classifier = InferenceEngineClassifier(config_path=model, weights_path=weights,
                                              device=device, extension=cpu_extension,
                                              classes_path=classes_path)
    top_n = 3
    result = []
    for img in to_classify:
        prob = ie_classifier.classify(img)
        predictions = ie_classifier.get_top(prob, top_n)
        predictions = [str(ie_classifier.labels_map[predictions[i]]) + ': '
                       + str(predictions[i]) + "  with confidence "
                       + str(prob[0][predictions[i]]) for i in range(top_n)]
        result.append(predictions)
        #log.info("Predictions: " + str(predictions))
    ranked = 0
    for prediction in result:
        print('Dog #{}:'.format(ranked+1))
        print("Top predictions:")
        for breed in result[ranked]:
            print('\t' + str(breed))
        ranked += 1
    #for i in range(result.shape):

    return result


def main():
    img1 = cv2.imread('img.png')
    img2 = cv2.imread('img_1.png')
    to_classify = [img1, img2]
    result = list_classifier(to_classify)

    return


if __name__ == '__main__':
    sys.exit(main())
