#!/bin/sh
# MNIST IDX files; md5 is checked in mnist_io.py against the checksums pinned in torchvision.
mkdir -p data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sO https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz; done
