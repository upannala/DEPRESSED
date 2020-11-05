# How to run this script
# python video.py --shape-predictor shape_predictor_68_face_landmarks.dat

# Importing relevant packages and libraries
import os
import sys
import cv2
import dlib
import numpy as np
import argparse
import time
import imutils
import shlex
import subprocess
import tensorflow as tf
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from imutils.video import FPS
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance as dist
from moviepy.editor import VideoFileClip
from moviepy.editor import *
import skvideo.io
import pyrebase
from firebase import firebase
import json
import boto3
import firebase_admin
from firebase_admin import credentials, firestore

#Bucket names 
BUCKET_NAME = 'dheergayu-objectdatastore'
BUCKET_NAME_CONFIGS = 'configsdhirghayu'

# Variables used to calculate depression rate
depressed=0
not_depressed=0
counter_frames=0
depression_rate=0

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
blink_rate=0
blink_depression=0

# Taking the video from S3 bucket
s3 = boto3.resource('s3')

#Downloading Libraries and Trained models from S3 bucket
bucket_configs = s3.Bucket(BUCKET_NAME_CONFIGS)
for obj in bucket_configs.objects.all():
    key = obj.key
    #body = obj.get()['Body'].read()
    s3.Bucket(BUCKET_NAME_CONFIGS).download_file(key, key)

#Initializing the firebase database
databaseURL = {
     'databaseURL': "https://dheergayuappclient.firebaseio.com"
}
cred = credentials.Certificate("creds.json")
firebase_admin.initialize_app(cred, databaseURL)
database_fs = firestore.client()
col_ref = database_fs.collection('report')

# Reading the Firebase records and updating the values
results = col_ref.order_by('date',direction='DESCENDING').get() 
for item in results:
    print(item.to_dict())
    print(item.id)
    doc = col_ref.document(item.id) # doc is DocumentReference
    field_updates = {"dynamic_comp": "NA"}
    result=doc.update(field_updates)

#Method that return the EAR
def eye_aspect_ratio(eye):
    # computing the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # computing the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # computing the EAR value
    ear = (A + B) / (2.0 * C)

    return ear


detector = dlib.get_frontal_face_detector()
#load model
model = model_from_json(open("model.json", "r").read())
#load weights
model.load_weights('model.h5')


#importing the haarcascade_frontalface_default.xml library
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Runtime arguements
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib face detector 
# Creating the facial landmark predictor
print("[INFO] loading facial landmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Facial landmark indexes for the left eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# Facial landmark indexes for the right eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


#Downloading Videos and start processing
bucket = s3.Bucket(BUCKET_NAME)
for obj in bucket.objects.all():
    key = obj.key
    
    #downloading the video from S3 bucket
    s3.Bucket(BUCKET_NAME).download_file(key, key)
    time.sleep(10)
    print("Video =",key)
    key_id=key.replace(".mp4", "")
    results = col_ref.order_by('date',direction='DESCENDING').get() 
    for item in results:
        if item.id == key_id and '.mp4' in key:
            # Starting the Video stream 
            cap=FileVideoStream(key).start()
            fps = FPS().start()
            fileStream = True
            time.sleep(1.0)

            #Identifying the video clip duration
            clip = VideoFileClip(key)
            clip_duration=(clip.duration)

            #Loop over frames of the Video provided            
            while True: 

                #Breaking the loop when there are no any frames
                if fileStream and not cap.more():
                    break
                before_rotate=cap.read()
                
                if before_rotate is None:
                    break
                
                #Resizing the image
                test_img = imutils.resize(before_rotate, width=450)
                
                #Converting it to grayscale
                gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

                faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
                
                #Detect the faces that are in the grayscale image
                rects = detector(gray_img, 0)
                

                for (x,y,w,h) in faces_detected:
                    
                    roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                    roi_gray=cv2.resize(roi_gray,(48,48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                    img_pixels /= 255

                    predictions = model.predict(img_pixels)

                    #Indentifiyng the max indexed 
                    max_index = np.argmax(predictions[0])

                    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                    predicted_emotion = emotions[max_index]

                    #Calculating the depressed emotion count
                    if predicted_emotion in('angry' ,'disgust' ,'fear' ,'sad'):
                        depressed = depressed +1
                    #Calculating the non depressed emotion count
                    else:
                        not_depressed = not_depressed + 1
                    
                    #Calculating the total number of frames
                    counter_frames=counter_frames+1
                    
                    #Updating the emotion depression level status
                    depression_rate=(100*depressed)/counter_frames
                    
            
                #Looping the face detections    
                for rect in rects:
                    
                    shape = predictor(gray_img, rect)
                    shape = face_utils.shape_to_np(shape)

                    # coordinates of left eye
                    leftEye = shape[lStart:lEnd]
                    # coordinates of right eye
                    rightEye = shape[rStart:rEnd]
                    # EAR of left eye
                    leftEAR = eye_aspect_ratio(leftEye)
                    # EAR of Right eye
                    rightEAR = eye_aspect_ratio(rightEye)

                    # average EAR value
                    ear = (leftEAR + rightEAR) / 2.0

                    # Convex hull for the left eye
                    leftEyeHull = cv2.convexHull(leftEye)
                    # Convex hull for the right eye
                    rightEyeHull = cv2.convexHull(rightEye)                   
                   
                    # Increment the counter if the EAR value is below the eye
                    # Threshhold value
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1

                    else:
                        # If the eye is closed for the required frames count
                        # increment the number of blinks
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1

                        # reseting the counter
                        COUNTER = 0
                                        
                    
                resized_img = cv2.resize(test_img, (1000, 700))
                
                fps.update()
                if cv2.waitKey(10) == ord('q'):
                    break

            print("[Info] No of blinks =",TOTAL) 
            #Calculating the blink depression rate       
            blink_rate=(TOTAL/clip_duration)*60
            if blink_rate<10.5:
                blink_depression=((10.5-blink_rate)/10.5)*100
            elif blink_rate>32:
                blink_depression=((blink_rate-32)/32)*100

            #Printing the information
            print("[Info] Total Frames==",counter_frames)
            print("[Info] Depressed Frames==",depressed)
            print("[Info] Non-Depressed Frames==",not_depressed)
            print("[Info] Blink Rate==",blink_rate)
            print("[Info] Emotion depression Rate==",depression_rate)
            print("[Info] Blink depression Rate==",blink_depression)
            print("[Info] elasped time:",clip_duration)
            
            fps.stop()
            cv2.destroyAllWindows
                
            #Updating the database
            doc = col_ref.document(key_id) 
            field_updates = {"dynamic_blink": blink_depression,
                            "dynamic_emotion": depression_rate,
                            "dynamic_comp": "YES"}
            result=doc.update(field_updates)
            
            print("[Info] DB Updated:",result)
            print("==========================================================") 
            
            #Resetting values for the next run
            COUNTER = 0
            TOTAL = 0
            blink_rate=0
            blink_depression=0 
            depressed=0
            not_depressed=0
            counter_frames=0
            s3.Object(BUCKET_NAME, key).delete()
        
        else :
            print("==========================================================") 
            