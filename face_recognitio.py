import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import keras
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pandas as pd

from tkinter import *



# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fr_model=load_model("facenet_keras.h5")

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    else:

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cropped_face = img[y:y + h, x:x + w]
        cv2.imshow("image",img)
        return cropped_face


# Initialize Webcam
def save_image():

    cap = cv2.VideoCapture(0)
    count = 0
    name=input("Enter Name of the person")
    while True:
        _, frame = cap.read()
        face = face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (160, 160))
            im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
            #img_array = np.expand_dims(img_array, axis=0)
            #encd = create_embeddings(img)

        # Save file in specified directory with unique name
            file_name_path = 'images' + str(name) + '.jpg'
            cv2.imwrite(file_name_path, face)
            #create_embeddings(file_name_path, name)
            encd = create_embeddings(face)
            print(encd)
            cap.release()
        # Put count on images and display live count
            #cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
            break


        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            print("Face not found")
            pass


   # cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")
    return encd
    #create_embeddings(file_name_path,name)

def create_embeddings(img):
    #image=cv2.imread(file_name_path)
    #img_resized=cv2.resize(image,(160,160))
  # face_pixels=np.asarray(img_resized)
    face_pixels = img.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = fr_model.predict(samples)
    return yhat





    #with open('persons.csv', 'a') as csvfile:
        #filewriter = csv.writer(csvfile, delimiter=',',
                                #quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #filewriter.writerow([name, encd])




#save_image()

def compare_img():
    img=cv2.imread("yourfaceimage.jpg")
    encd=create_embeddings(img)
    video_capture = cv2.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        # canvas = detect(gray, frame)
        # image, face =face_detector(frame)

        face = face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (160, 160))
            #im = Image.fromarray(face, 'RGB')
            # Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(face)
            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            #img_array = np.expand_dims(img_array, axis=0)
            pred = create_embeddings(img_array)


            dist=np.linalg.norm(encd-pred)
            #print(pred.shape)
            print(dist)
            if(dist < 13):
                cv2.putText(frame, "Reynold", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unkown", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

#compare_img()
root=Tk()
root.geometry('200x150')
b1=Button(root,text="BEGIN",height="10",width="15",command=compare_img,bg="yellow",fg="blue")
b1.pack(side = TOP, pady = 5)
root.mainloop()

