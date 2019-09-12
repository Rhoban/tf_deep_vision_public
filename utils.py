#!/usr/bin/env python

import sys
import random
import os
import cv2
import numpy as np
import json
def getJson(file):
        return file[:-len('.png')]+'.json'

def getImg(file):
        return file[:-len('.json')]+'.png'

# Taken from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
            
def shuffleDict(d):
        keys = list(d.keys())
        random.shuffle(keys)

        ret = {}
        
        for k in keys:
                if(k not in ret):
                        ret[k] = []
                for v in d[k]:
                        ret[k].append(v)
                
        return ret


def visualize_false_predictions(model_folder, batch_test_images, batch_test_labels, batch_test_paths, sample, inverted_labels_map):
        false_predictions = []
        
        imageIndex = 0
        os.system("mkdir "+str(model_folder)+"/visualization/")
        for b in range(0, len(sample)):
                # visual evaluation
                tmpIm = batch_test_images[b]
                tmpIm = cv2.resize(tmpIm, (64, 64))
                image_size = [tmpIm.shape[0], tmpIm.shape[1]]
                true = batch_test_labels[b]
                pred = sample[b]
                if (np.argmax(true) != np.argmax(pred)):
                        blank_image = np.zeros((image_size[1]*2, image_size[0]*2, 3))
                        blank_image[:, :, 0] = 255
                        blank_image[:, :, 1] = 255
                        blank_image[:, :, 2] = 255
                        blank_image[0:image_size[1], 0:image_size[0], :] = (tmpIm*255).astype(np.uint8)
                        
                        str_true = "true : "+inverted_labels_map[str(np.argmax(true))]
                        str_pred = "pred : "+inverted_labels_map[str(np.argmax(pred))]
                        
                        cv2.putText(blank_image, str_true, (0, image_size[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(blank_image, str_pred, (0, image_size[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imwrite(str(model_folder)+"/visualization/false_prediction_"+str(imageIndex)+".png", blank_image.astype(np.uint8))

                        false_predictions.append({"path" : batch_test_paths[b], "true" : str(inverted_labels_map[str(np.argmax(true))]), "pred": str(inverted_labels_map[str(np.argmax(pred))])})
                        
                        imageIndex +=1


        with open(str(model_folder)+"/visualization/false_predictions.json", 'w') as f:
                json.dump(false_predictions, f)


def create_prediction_image(im, label):
        im = cv2.resize(im, (64, 64))
        image_size = [im.shape[0], im.shape[1]]        

        blank_image = np.zeros((image_size[1]*2, image_size[0]*2, 3))
        blank_image[:, :, 0] = 255
        blank_image[:, :, 1] = 255
        blank_image[:, :, 2] = 255
        blank_image[0:image_size[1], 0:image_size[0], :] = (im*255).astype(np.uint8)
        cv2.putText(blank_image, label, (0, image_size[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        return blank_image
            

def cleanModels():
    os.system("rm -rf model_*")
    os.system("unlink lastModel")

def softmax(x):
        """
        Compute softmax values for each sets of scores in x.
    
        Rows are scores for each class. 
        Columns are predictions (samples).
        """
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)
