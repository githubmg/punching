# This is the base Simon Python Script that will work without 
# needing Deeplens deployment.

import os
#import greengrasssdk
from threading import Timer
import scipy
import time
import numpy as np
import awscam
import cv2
import json
import requests
import shutil
import os
import mxnet as mx
from threading import Thread
#from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from pprint import pprint


#from pygame import mixer

### Helper Functions ##
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h==184) else 184-h # down
    pad[3] = 0 if (w==184) else 184-w # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

## Greengrass Loop ##
def greengrass_infinite_infer_run():
    #try:
        #game = SimonGame('test')
        ##TODO FIX THIS PATH
        modelPath = "/home/aws_cam/faster_184.xml"
        
        # Send a starting message to IoT console
        #client.publish(topic=iotTopic, payload="Simon Say Game Starting")
        #results_thread = FIFO_Thread()
        #results_thread.start()

        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(modelPath, mcfg)
        #client.publish(topic=iotTopic, payload="Model loaded")

        doInfer = True
        #game_count = 0
        poses = []
        collect_data = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            
            
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")
            
            
            img = Image.fromarray(frame, 'RGB')
            img.save('frame.png')
            
            
            #Prepare Image for Network
            
            print frame.shape
            center = frame.shape[1]/2
            left = center - (frame.shape[0]/2)
            scale = frame.shape[0]/184
            offset = (frame.shape[1] - frame.shape[0]) / 2
            
            cframe = frame[0:1520,left:left+1520,:]
            scaledImg = image_resize(cframe, width=184)
            
            img = Image.fromarray(scaledImg, 'RGB')
            img.save('scaledImg.png')
            
            
            heatmap_avg = np.zeros((scaledImg.shape[0], scaledImg.shape[1], 16))
            paf_avg = np.zeros((scaledImg.shape[0], scaledImg.shape[1], 28))

            imageToTest = cv2.resize(scaledImg, (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
            print imageToTest.shape
            
            imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 128)
            
            img = Image.fromarray(imageToTest_padded, 'RGB')
            img.save('imageToTest_padded.png')
            
            print pad
            transposeImage = np.transpose(np.float32(imageToTest_padded[:,:,:]), (2,0,1))/255.0-0.5

           
            startt = time.time()
            output = model.doInference(transposeImage)
            
            endt = time.time()
            print (endt - startt)
            
            h = output["Mconv7_stage4_L2"]
            print h.shape
            p = output["Mconv7_stage4_L1"]
            print p.shape
            
            heatmap1 = h.reshape([16,23,23]) 
            heatmap = np.transpose(heatmap1, (1,2,0))

            #print heatmap1.shape
            
            #heatmap = np.moveaxis(h, 0, -1)

            heatmap = cv2.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv2.resize(heatmap, (scaledImg.shape[1], scaledImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            print 'heatmapshape'
            print heatmap.shape
            
            all_peaks = []
            peak_counter = 0
            
            param={}
            param['octave'] = 3
            param['use_gpu'] = 1
            param['starting_range'] = 0.8
            param['ending_range'] = 2
            param['scale_search'] = [0.5, 1, 1.5, 2]
            param['thre1'] = 0.1
            param['thre2'] = 0.05
            param['thre3'] = 0.5
            param['mid_num'] = 4
            param['min_num'] = 10
            param['crop_ratio'] = 2.5
            param['bbox_ratio'] = 0.25
            param['GPUdeviceNumber'] = 3
            
            for part in range(17-1):
                x_list = []
                y_list = []
                map_ori = heatmap_avg[:,:,part]
                map = gaussian_filter(map_ori, sigma=3)

                map_left = np.zeros(map.shape)
                map_left[1:,:] = map[:-1,:]
                map_right = np.zeros(map.shape)
                map_right[:-1,:] = map[1:,:]
                map_up = np.zeros(map.shape)
                map_up[:,1:] = map[:,:-1]
                map_down = np.zeros(map.shape)
                map_down[:,:-1] = map[:,1:]

                peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
                peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
                peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
                id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)
            
            print 'allpeaks'
            pprint(all_peaks)
            
            

            
#    except Exception as e:
#        msg = "Test failed: " + str(e)
#        print e
#client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    #Timer(15, greengrass_infinite_infer_run).start()

# Execute the function above
greengrass_infinite_infer_run()

# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
