#!/usr/bin/env python
# coding:utf-8

import caffe
import numpy as np
import sys
import scipy.io

# convert mean file (.mat) to .binaryproto
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_mat",
        help="input a mat file"
    )
    parser.add_argument(
        "output",
        help="output .binaryproto"
    )

    args = parser.parse_args()
    BINARY_PROTO_FILE_NAME = args.output
    mat_dict = scipy.io.loadmat(args.input_mat)
    image_mean = mat_dict['image_mean']

    image_mean = np.transpose(image_mean, (2, 0, 1))
    image_mean = image_mean.reshape((1, image_mean.shape[0], image_mean.shape[1], image_mean.shape[2]))

    blob = caffe.io.array_to_blobproto(image_mean)

    data = blob.SerializeToString()
    open(BINARY_PROTO_FILE_NAME, 'wb').write(data)
