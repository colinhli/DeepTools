#!/usr/bin/env python
# coding:utf-8
import os
import sys
from PIL import Image, ImageDraw
import shutil  
import numpy as np
import argparse
import sys
sys.path.append('./caffe/python')
import caffe
import math

#wrap _caffe.blob
class CaffeBlob(object): 
    def __init__(self, net, blobname):
        self.net = net
        self.blobname = blobname
        self.channels = self.net.blobs[self.blobname].channels
        self.width = self.net.blobs[self.blobname].width
        self.height = self.net.blobs[self.blobname].height
        self.data = self.net.blobs[self.blobname].data 

    def get_image(self):  
        M = 0
        N = 0
        if math.floor(math.sqrt(self.channels)) ** 2 != self.channels:
            N = math.ceil(math.sqrt(self.channels))
            while self.channels % N != 0 and N < 1.2 * math.sqrt(self.channels):
                N = N + 1

            M = math.ceil(self.channels / N)
        else:
            N = math.sqrt(self.channels)
            M = N

        M = int(M)
        N = int(N)
        print 'features map: {0} * ({1} * {2})'.format(self.channels, M, N) 
        
        margin = 4
        image_size = self.width
        features_image = np.zeros((margin + M * (image_size + margin), margin + N * (image_size + margin)), dtype=np.uint8)
        features_image[:, :] = 255
        for i in xrange(self.channels):
            d =  self.data[0, i, :, :]
            image = d.copy()
            image -= d.min()
            image /= (d.max() - d.min())
            image *= 255 
            image = image.astype(np.uint8)

            m = int(i / N)
            n = int(i % N)
            features_image[margin - 1 + m * (image_size + margin): margin - 1 + m * (image_size + margin) + image_size,
                        margin - 1 + n * (image_size + margin): margin - 1 + n * (image_size + margin) + image_size] = image

        return Image.fromarray(features_image)

class Feature(caffe.Classifier):
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Classifier.__init__(self, model_file, pretrained_file, image_dims = image_dims)
    def set_weights():
        pass
    def extract_feature(self, input, layer_name): 
        in_ = caffe.io.resize_image(input, self.image_dims) 
        caffe_in = np.zeros((1, input.shape[2], self.image_dims[1], self.image_dims[0]), dtype=np.float32) 
        caffe_in[0] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})   
        blob = CaffeBlob(self, layer_name) 
        blob.get_image().show()

class VGGFeature(Feature):
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):

        Feature.__init__(self, model_file, pretrained_file, image_dims,
                 mean, input_scale, raw_scale,
                 channel_swap)
        self.set_weights()

    def set_weights(self):
        pass
    
def create_vgg_mean():
    mean = np.zeros((3, 224, 224), dtype=np.float32)
    mean[0, :, :] =  104.008 
    mean[1, :, :] =  116.669
    mean[2, :, :] =  123.68 
    np.save('./models/VGG/vgg_mean.npy', mean)

def main(argv):
    pycaffe_dir = os.path.dirname(__file__) 
    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "input_file",
        help="Input txt/csv filename. If .txt, must be list of filenames.\
        If .csv, must be comma-separated file with header\
        'filename, xmin, ymin, xmax, ymax'"
    )

    # parser.add_argument(
    #     "output_file",
    #     help="Output h5/csv filename. Format depends on extension."
    # )
    # Optional arguments.
    
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/VGG/VGG_ILSVRC_16_layers_depoly.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/VGG/VGG_ILSVRC_16_layers.caffemodel"),
        help="Trained model weights file."
    )  
   
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV." 
    )

    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             '../models/VGG/vgg_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    
    args = parser.parse_args() 
    mean = np.load(args.mean_file)  
    image = caffe.io.load_image(args.input_file)  

    detector = VGGFeature(args.model_def, args.pretrained_model, mean=mean,
            raw_scale=255.0, channel_swap='2,1,0') 
    
    
    detector.extract_feature(image, 'conv1_1')

if __name__ == "__main__": 
    main(sys.argv)