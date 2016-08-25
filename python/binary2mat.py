#!/usr/bin/env python
# coding:utf-8
import caffe
import numpy as np
import sys
import scipy.io
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "binaryproto",
        help="transformed binaryproto file"
    )
    parser.add_argument(
        "output",
        help="output .mat"
    )

    args = parser.parse_args()
    BINARY_PROTO_FILE_NAME = args.binaryproto

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(BINARY_PROTO_FILE_NAME, 'rb').read()
    blob.ParseFromString(data)

    arr = np.array(caffe.io.blobproto_to_array(blob)) 
    out = arr[0]
    out = np.transpose(out, (1, 2, 0))
    scipy.io.savemat(args.output, dict(image_mean=out))
