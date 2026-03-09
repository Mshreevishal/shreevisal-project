import cv2
import os
import numpy as np
from PIL import Image

def train():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces=[]
    ids=[]

    for file in os.listdir("dataset"):

        path = os.path.join("dataset",file)

        img = Image.open(path).convert('L')

        img_np = np.array(img,'uint8')

        student_id = int(file.split(".")[1])

        faces.append(img_np)
        ids.append(student_id)

    recognizer.train(faces,np.array(ids))

    if not os.path.exists("trainer"):
        os.makedirs("trainer")

    recognizer.save("trainer/model.yml")

    print("Training Completed")