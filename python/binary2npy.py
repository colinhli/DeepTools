#!/usr/bin/env python
#coding:utf-8 
import numpy as np
import sys
import scipy.io
import argparse
sys.path.append("./caffe/python")
import caffe
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "binaryproto",
        help="transformed binaryproto file"
    )
    parser.add_argument(
        "output",
        help="output .npy [Channels x Height x Width] "
    )

    args = parser.parse_args()
    BINARY_PROTO_FILE_NAME = args.binaryproto
    NPY_FILE_PATH = args.output 
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(BINARY_PROTO_FILE_NAME, 'rb').read()
    blob.ParseFromString(data)

    arr = np.array(caffe.io.blobproto_to_array(blob)) 
    out = arr[0]
    # out = np.transpose(out, (2, 1, 0))
    np.save(NPY_FILE_PATH, out)

