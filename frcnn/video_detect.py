import numpy as np
import time
import os
import sys
import json
import argparse
from random import randint
import urllib
import logging
import datetime
import redis
import pickle

import cv2
import gluoncv as gcv
import mxnet as mx
from utils import return_tags


from azure import *
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings
from azure.storage.blob import (
    BlockBlobService,
    ContainerPermissions,
    BlobPermissions
)

storage_account_name = configData['storage_account_name']
storage_account_key = configData['storage_account_key']
storage_container = configData['storage_container']

redisHost= configData['redisHost']
redisPort= configData['redisPort']
storage_block_blob_service = BlockBlobService(account_name=storage_account_name, account_key=storage_account_key)
redisConnection = redis.StrictRedis( host = redisHost, port = redisPort, db = 0)

def tagger(net, video_path, frames_process=20): 
    # Load the video
    video = cv2.VideoCapture(video_path)
    time.sleep(1) ### letting the camera autofocus
    found_tags = []
    print("STARTING")
    try:
        frame_id = -1
        ret = True
        while(video.isOpened()):
            ret, frame = video.read()
            print("GOT FRAME")
            print(ret)
            frame_id += 1
            if frame_id % frames_process != 0:
                print(frame_id)
                continue 
            else:
                if ret == False:
                    break

                ## Preprocess the img
                frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
                rgb_nd, frame = gcv.data.transforms.presets.rcnn.transform_test(frame, short=512, max_size=700)

                class_ids, scores, bboxes = net(rgb_nd.as_in_context(mx.gpu(0)))

                found_tags.append(return_tags(rgb_nd, bboxes[0], scores[0], class_ids[0], class_names=net.classes))

            if ret == False:
                break

        print("RETURNING FOUND TAGS")
        return found_tags

    except Exception as err:
        print(err)
    finally:
        #print(found_tags)
        video.release()
        cv2.destroyAllWindows()
        print("DONE")

# Load the model
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))
#net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=mx.gpu(0))
#net = gcv.model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True, ctx=mx.gpu(0))

#for i in range(1,5):
#    print(tagger(net, './20-26-20.mp4', 20))
#print(tagger(net, './rickgarage.mp4', 20))

def print_phase_header(message):
    global COUNTER;
    print ("\n[" + str("%02d" % int(COUNTER)) + "] >>> " +  message)
    COUNTER += 1;

def print_phase_message(message):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print (str(time_stamp) + ": " +  message)

## Doesn't seem like framesProcess is being passed as argument, hardcoded instead to 20
ap = argparse.ArgumentParser()
ap.add_argument("-cf", "--cameraframes", type=int, default=10, help="process every n frames")
args = vars(ap.parse_args())

index = 0
slotId = 0
framesProcess = 20
net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=mx.gpu(0))

while ifTrue:
    if index == 500:
       break
    if os.path.isfile("die.txt"):
       print( "BYE BYE " + str( slotId))
       break
    chance=randint(0, 9)
    if chance == 1:
        pop=redisConnection.rpop
    else:
        pop=redisConnection.lpop
#nextItemOrig = pop( "indexer_queue_" + suffix)
#    nextItemOrig = pop( "indexer_queue_" + 'ml')
    nextItemOrig = pop( "indexer_queue")
    if nextItemOrig == None or nextItemOrig == '':
        print_phase_message( "Sleeping, nothing to do...")
#        break
        time.sleep(60)
        continue
#    index=index + 1
    nextItem = nextItemOrig.decode("utf-8")
#    print( "NEXTITEM " + nextItem)
    if nextItem == None or nextItem=='':
        print_phase_message( "Sleeping, nothing to do...")
#        break
        time.sleep(60)
        continue

    index=index + 1
    parts = nextItem.split(":")
    eventId=parts[0]
#    eventId = str( int( eventId) + 1)  ####REMOVE
    videoPath='/'.join( parts[1].split('/')[1:])
    local_video_path = "./result/" + videoPath.replace("/", "_")
    videoID = str( eventId) + ":" + videoPath
    print( "nextItem " + nextItem + "  " + local_video_path)
    try:
        downloadBlob=storage_block_blob_service.get_blob_to_path( storage_container, videoPath, local_video_path)
    except:
        print( "ERROR DOWNLOADING")

    print( index, eventId, videoPath)
    try:
        foundLabels = tagger(net, local_video_path, framesProcess)
        redisConnection.rpush( "result_queue_ml", videoID )
        redisConnection.rpush( "ml:result_queue:" + suffix, suffix + ":" + videoID )
        redisConnection.set( suffix + ":" + videoID, foundLabels, 864000)
        redisConnection.sadd( "stat_ml_set:" + videoID, suffix)
        redisConnection.expire( "stat_ml_set:" + videoID, 864000)
        print( foundLabels)
    except:
        print( "ERROR PROCESSING")

    try:
#    	pass
        os.remove( local_video_path)
    except:
        print( "Removed failed" + local_video_path)
