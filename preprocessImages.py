#!/usr/bin/env python

import cv2
import os
import math
import utils

dataPaths = []
badImages = []

for dataPath in dataPaths:
    for file in os.listdir(dataPath):
            if file.endswith(".png"):
                im = cv2.imread(dataPath+"/"+file)
                if im is None:
                    badImages.append(dataPath+"/"+file)

if len(badImages)>0:
    print("Detected "+str(len(badImages))+" bad images")

    answer = utils.query_yes_no("Delete bad images ?", default="no")
    if answer is True:
        for b in badImages:
            print("DELETING "+b)
            os.system("rm "+b)
            os.system("rm "+utils.getJson(b))

else:
    print("No bad images detected !")

    
            
