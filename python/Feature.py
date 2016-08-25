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

class Feature(caffe.Classifier):
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Classifier.__init__(self, model_file, pretrained_file, image_dims = image_dims)

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

    detector = Feature(args.model_def, args.pretrained_model, mean=mean,
            raw_scale=255.0, channel_swap='2,1,0') 
    image = caffe.io.resize_image(image, detector.image_dims) 
    
    
if __name__ == "__main__": 
    main(sys.argv)