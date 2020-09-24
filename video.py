# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4

# Importing relevant packages and libraries
import os
import cv2
import dlib
import numpy as np
import argparse
import time
import imutils
import shlex
import subprocess
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
from firebase import firebase
import json

# Variables used to calculate depression rate
depressed=0
not_depressed=0
counter_frames=0
depression_rate=0

EYE_AR_THRESH = 0.3#0.275
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
blink_rate=0
blink_depression=0

#Method that return the EAR
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


detector = dlib.get_frontal_face_detector()
#load model
model = model_from_json(open("model.json", "r").read())
#load weights
model.load_weights('model.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Runtime arguements
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
    help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#cap=cv2.VideoCapture(0)
# Taking the video

cap=FileVideoStream(args["video"]).start()
fps = FPS().start()
fileStream = True
time.sleep(1.0)

clip = VideoFileClip(args["video"])
clip_duration=(clip.duration)


while True: 
    if fileStream and not cap.more():
        break
    before_rotate=cap.read()# captures frame and returns boolean value and captured image
    
    #print("image==",test_img)
    if before_rotate is None:
        print("Image is null ",test_img)
        break
    #if not ret:
     #   continue
    #test_img = imutils.rotate_bound(before_rotate, -90)
    
    scale_percent = 50 # percent of original size
    #print('Original Dimensions : ',test_img.shape)
    #width = int(test_img.shape[1] * scale_percent / 100)
    #height = int(test_img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    #test_img = cv2.resize(test_img, dim, interpolation = cv2.INTER_AREA)
    test_img = imutils.resize(before_rotate, width=450)
    #print('Resized Dimensions : ',test_img.shape)
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    rects = detector(gray_img, 0)
    #f not ret:
    #  continue
    #if ret:
    #    assert not isinstance(test_img,type(None)), 'frame not found'
    


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        #cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if predicted_emotion in('angry' ,'disgust' ,'fear' ,'sad'):
            depressed = depressed +1
        else:
            not_depressed = not_depressed + 1
        
        counter_frames=counter_frames+1
        #print("counter frames==",counter_frames)
        depression_rate=(100*depressed)/counter_frames
        #print("Not depressed==",not_depressed)
        #print("Depressed==",depressed)
        print("Rate==",depression_rate)
   
        
    for rect in rects:
        
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray_img, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(test_img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(test_img, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # reset the eye frame counter
            COUNTER = 0
        ##print("Blink Count==",TOTAL)
        
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(test_img, "Blinks: {}".format(TOTAL), (10, 30),#
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#
        cv2.putText(test_img, "EAR: {:.2f}".format(ear), (300, 30),#
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)#


    fps.update()
    if cv2.waitKey(10) == ord('q'):
        break


blink_rate=(TOTAL/clip_duration)*60
if blink_rate<10.5:
    blink_depression=((10.5-blink_rate)/10.5)*100
elif blink_rate>32:
    blink_depression=((blink_rate-32)/32)*100
print("Blink Rate==",blink_rate)
#print("Height==",height)
#print("Width==",width)
print("Rate==",depression_rate)
print("Blink depression Rate==",blink_depression)
fps.stop()
print("[INFO] elasped time:",clip_duration)
##cap.release()#
cv2.destroyAllWindows

firebase = firebase.FirebaseApplication('https://dirghayu-f1a14.firebaseio.com/', None)  
data =  { 'Name': 'Udith',  
          'RollNo': depression_rate,  
          'Percentage': blink_depression  
          }  
#data =  json.dumps({'Rate': depression_rate, 'Blink depression Rate': blink_depression})
result = firebase.post('dirghayu-f1a14/Face/',data)  
print(result)  
