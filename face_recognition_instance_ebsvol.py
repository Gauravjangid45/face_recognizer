#!/usr/bin/env python
# coding: utf-8

# ## Collecting Dataset

# In[1]:


import numpy as np
import cv2
import time

# load haarcascade face classifier
face_classifier = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml" )
 
# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is () :
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        
    return cropped_face




# Collect 100 samples of your face from webcam input
def collect_samples(path, count):
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = face_extractor(frame)
#             face = cv2.resize(face_extractor(frame), (200, 200))

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
            # Save file in specified directory with unique name
            file_name_path = path + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
        
            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)
        
        else:
            print("Face not found")
            pass
        if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
            break
        
    cap.release()
    cv2.destroyAllWindows()      
    print("Dataset Collected")


i =0 
j=0

print("Collecting Samples Of Gaurav")
collect_samples("./Faces/gaurav/", i)

time.sleep(3)

print("Collecting Samples Of Friend")
collect_samples("./Faces/friend/", j)


# ## Traning Model

# In[2]:


from sys import path
import cv2
import numpy as np
from os import listdir
from os.path  import isfile, join
# Get training data we previously made
def trainModel(path):
    
    data_path = path
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # Create arrays from training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    
    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    # model = cv2.face.createLBPHFaceRecognizer()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # pip install opencv-contrib-python
    # model = cv2.createLBPHFaceRecognizer()
    
    model = cv2.face_LBPHFaceRecognizer.create()
    # Let's train our model 
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model
  


gaurav_model = trainModel("./Faces/gaurav/")
print("Gaurav's_model trained sucessefully")
friend_model = trainModel("./Faces/friend/")
print("Friend's_model trained sucessefully")


# In[4]:


import pywhatkit


# ## Run our Face Recognition

# In[5]:


# from TrainModel import aditya_model, friend_model 
import cv2
import numpy as np
import os

import subprocess

def confidence(results, image):
    if results[1] < 500 :
        confidence = int( 100 * (1 - (results[1])/400) )
        display_string = str(confidence) + '% Confident that the user is'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
    return confidence

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is () :
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# Open Webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        gaurav_results = gaurav_model.predict(face)
        friend_results = friend_model.predict(face)
        
        c_gaurav = confidence(gaurav_results, image)
        c_friend = confidence(friend_results, image)

        if c_gaurav > 90:
            cv2.putText(image, "Gaurav", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            try:
#                 print("you are caught up guarav")
                  pywhatkit.sendwhatmsg_instantly('+919461485431', 'Face Recognition successfully done')
                  print('!! Message Sent Successfully !!')
                  pywhatkit.send_mail('gauravpkt04@gmail.com',
                                 'Second@123',
                                 'Face Recognition Testing',
                                 'This is the face of Gaurav jangid....',
                                 'gauravjangidamer@gmail.com')
            except: 
                    print("!! Allow less secure apps in Email settings !! ")
                    break
         
        elif c_friend > 90:
            cv2.putText(image, "Friend", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,87,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            try:
#                 print("your friend also caught up")

#                 os.system("aws ec2 run-instances --image-id ami-0ad704c126371a549 --count 1 --instance-type t2.micro --key-name gaurav --security-group-ids sg-0c6fb189befe44ec0")
                instance = subprocess.getoutput("aws ec2 run-instances --image-id ami-0ad704c126371a549 --count 1 --instance-type t2.micro --key-name gaurav --security-group-ids sg-0c6fb189befe44ec0")
                volume = subprocess.getoutput("aws ec2 create-volume --size 1 --availability-zone ap-south-1a")
                subprocess.getoutput(f"aws ec2 attach-volume --volume-id {volume[191:212]} --instance-id {instance[157:176]} --device /dev/sdf")
#                 os.system("aws ec2 create-volume --size 1 --availability-zone ap-south-1b")
#                 os.system("aws ec2 attach-volume --volume-id vol-017d27cf6a9d3ddc7 --instance-id i-0125d859e79f73624 --device /dev/sdf")
                print('!! EC2 Launched... !!\n!! 5GB EBS Volume Created and Attached... !!')
                break
            except:
                print("!! Error...... !!")
                break

        else:    
            cv2.putText(image, "Rcognizing Face...", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
        
    except:
        cv2.putText(image, "No Face Found!", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Looking For Face...", (220, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      


# In[ ]:




