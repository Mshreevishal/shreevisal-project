import cv2
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# folders
if not os.path.exists("TrainingImage"):
    os.makedirs("TrainingImage")

if not os.path.exists("TrainingImageLabel"):
    os.makedirs("TrainingImageLabel")

if not os.path.exists("Attendance"):
    os.makedirs("Attendance")


###############################################
# Capture Images
###############################################
def take_images():

    Id = txt_id.get()
    name = txt_name.get()

    if Id == "" or name == "":
        messagebox.showerror("Error","Enter ID and Name")
        return

    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    sampleNum = 0

    while True:

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            sampleNum += 1

            cv2.imwrite(
                "TrainingImage/"+name+"."+Id+"."+str(sampleNum)+".jpg",
                gray[y:y+h,x:x+w]
            )

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Capturing Faces",img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elif sampleNum > 60:
            break

    cam.release()
    cv2.destroyAllWindows()

    row = [Id,name]

    if not os.path.exists("StudentDetails.csv"):
        with open("StudentDetails.csv","w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Id","Name"])

    with open("StudentDetails.csv","a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    messagebox.showinfo("Success","Images captured successfully")


###############################################
# Train Model
###############################################
def train_images():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces=[]
    Ids=[]

    for imagePath in os.listdir("TrainingImage"):

        path = os.path.join("TrainingImage",imagePath)

        img = Image.open(path).convert('L')
        img_numpy = np.array(img,'uint8')

        Id=int(os.path.split(path)[-1].split(".")[1])

        faces.append(img_numpy)
        Ids.append(Id)

    recognizer.train(faces,np.array(Ids))
    recognizer.save("TrainingImageLabel/trainer.yml")

    messagebox.showinfo("Success","Training Completed")


###############################################
# Track Attendance
###############################################
def track_images():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/trainer.yml")

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    )

    df = pd.read_csv("StudentDetails.csv")

    cam = cv2.VideoCapture(0)

    font=cv2.FONT_HERSHEY_SIMPLEX

    attendance=[]

    while True:

        ret,img = cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,1.2,5)

        for(x,y,w,h) in faces:

            Id,conf = recognizer.predict(gray[y:y+h,x:x+w])

            if conf < 50:

                name=df.loc[df["Id"]==Id]["Name"].values[0]

                ts=datetime.now()
                date=ts.strftime("%Y-%m-%d")
                time=ts.strftime("%H:%M:%S")

                attendance=[Id,name,date,time]

                cv2.putText(img,str(name),(x,y-10),font,1,(0,255,0),2)

            else:
                cv2.putText(img,"Unknown",(x,y-10),font,1,(0,0,255),2)

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Attendance",img)

        if cv2.waitKey(1)==ord('q'):
            break

    ts=datetime.now()
    fileName="Attendance/Attendance_"+ts.strftime("%Y-%m-%d")+".csv"

    if attendance:
        with open(fileName,"a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(attendance)

    cam.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Done","Attendance Saved")


###############################################
# UI
###############################################
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("500x400")
window.configure(bg="#2c3e50")

title=tk.Label(window,text="Attendance System",
               font=("Arial",20,"bold"),
               bg="#2c3e50",
               fg="white")
title.pack(pady=20)

lbl1=tk.Label(window,text="Student ID",bg="#2c3e50",fg="white")
lbl1.pack()

txt_id=tk.Entry(window,width=30)
txt_id.pack(pady=5)

lbl2=tk.Label(window,text="Student Name",bg="#2c3e50",fg="white")
lbl2.pack()

txt_name=tk.Entry(window,width=30)
txt_name.pack(pady=5)

btn1=tk.Button(window,text="Capture Images",
               command=take_images,
               width=20,
               bg="#3498db",
               fg="white")
btn1.pack(pady=10)

btn2=tk.Button(window,text="Train Model",
               command=train_images,
               width=20,
               bg="#27ae60",
               fg="white")
btn2.pack(pady=10)

btn3=tk.Button(window,text="Start Attendance",
               command=track_images,
               width=20,
               bg="#e74c3c",
               fg="white")
btn3.pack(pady=10)

window.mainloop()