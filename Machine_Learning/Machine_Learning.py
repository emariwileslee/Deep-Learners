import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import os
import yolo
from yolo.frontend import create_yolo
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations

def detector(image_loc):
    # 1. create yolo instance
    yolo_detector = create_yolo("ResNet50", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)
    
    # 2. load pretrained weighted file
    DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "weights.h5")
    yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)
      
    # 3. Load images
    DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "imgs")

    img_files = [os.path.join(DEFAULT_IMAGE_FOLDER, image_loc)]
    imgs = []
    for fname in img_files:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        #plt.imshow(img)
        #plt.show()
        
    # 4. Predict digit region
    THRESHOLD = 0.3
    for img in imgs:
        boxes, probs = yolo_detector.predict(img, THRESHOLD)
        
    return probs

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def get_number(prob_buffer):
    output_vector = first_nonzero(prob_buffer,axis=1,invalid_val=-1)
    output = ""
    
    for val in output_vector:
        output += str(val)
    return output

def check_if_correct(answer_value, detect):
    answer_len = len(answer_value)
    detection_len = len(detect)
    tracker = answer_len
    av_array = []
    detect_array = []

    for i in answer_value:
        av_array.append(int(i))

    for i in detect:
        detect_array.append(int(i))

    while tracker <= detection_len:
        perms = permutations(detect_array, tracker)
        for values in perms:
            if values == tuple(av_array):
                return int(answer_value)
        tracker += 1
    return int(detect) 
