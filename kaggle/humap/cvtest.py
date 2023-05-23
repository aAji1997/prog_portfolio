#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:39:35 2022

@author: hal
"""
from cv2 import cv2
print(cv2.__version__)

img = cv2.imread("./train_images/62.tiff")
window_name = "image"
cv2.imshow(window_name, img)
cv2.waitKey(1000)


