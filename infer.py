#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import *

from scipy.spatial.distance import cdist
from matplotlib import gridspec
import os
import cv2
import argparse
import utils
import json

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str

argParser = argparse.ArgumentParser(description='')
argParser.add_argument('-i', '--inputDir', type=typeDir, required=True, help="folder containing patchs")
argParser.add_argument('-o', '--outputDir', type=typeDir, required=True, help="output directory")
argParser.add_argument('-m', '--modelDir', type=typeDir, required=True, help="folder containing model")
argParser.add_argument('-e', '--epoch', type=str, required=False, help="epoch", default="lastEpoch")
argParser.add_argument('--imWidth', type=int, required=False, help="images width", default=16)
argParser.add_argument('--imHeight', type=int, required=False, help="images height", default=16)
args = argParser.parse_args()

images = []

for file in sorted(os.listdir(args.inputDir)):
        if file.endswith(".png"):
            im = cv2.imread(args.inputDir+"/"+file)
            im = cv2.resize(im, (args.imWidth, args.imHeight))
            im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)        
            images.append([im, file])


os.environ["CUDA_VISIBLE_DEVICES"]='1'

# Empty is always used, order is not important here
featuresUsed = [ "Ball"
                  ,"Empty"
                  ,"PostBase"
                  ,"Robot"
                  # ,"PenaltyMark"
                  ,"LineCorner"
                  # ,"ArenaCorner"
                  # ,"Center"
                  ,"T"
                  ,"X"
]
featuresUsed = sorted(featuresUsed)

placeholder_shape = [None] + list(images[0][0].shape)
img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
model_params = json.load(open(str(args.modelDir)+"/model_params.json"))


model = generic_cnn(img_placeholder, model_params)

results = {}

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, args.modelDir+"/"+str(args.epoch)+"/model.ckpt")
    
    print("Processing...")
    for i in range(0, len(images)):
        
        im = images[i][0]
        
        res = sess.run(model, feed_dict={img_placeholder:[im]})

        res = featuresUsed[np.argmax(res)]
        if res not in results:
            results[res] = 0

        results[res] += 1

        pred = utils.create_prediction_image(images[i][0], res)

        cv2.imwrite(args.outputDir+"/"+str(i)+".png", pred)

    print("DONE")
