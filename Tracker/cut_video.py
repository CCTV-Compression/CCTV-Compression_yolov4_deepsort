#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:53:46 2020

@author: hb
"""
import cv2
from PIL import Image

video_dir = '/home/hb/Desktop/yolov4-deepsort/data/video/test.mp4'
output_dir = '/home/hb/Desktop/yolov4-deepsort/outputs/test_small.mp4'
try:
    vid = cv2.VideoCapture(int(video_dir))
except:
    vid = cv2.VideoCapture(video_dir)
    

# by default VideoCapture returns float instead of int
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_dir, codec, fps, (width, height))

frame_id = 0
# while video is running
while True:
    return_value, frame = vid.read()
    out.write(frame)
    
    if frame_id == 499:
        break
    frame_id += 1