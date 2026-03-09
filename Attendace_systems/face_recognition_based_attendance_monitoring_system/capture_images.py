import cv2
import os
from database import add_student

def capture(student_id,name):

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    cam = cv2.VideoCapture(0)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0

    while True:

        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            count += 1

            path = f"dataset/{name}.{student_id}.{count}.jpg"

            cv2.imwrite(path,gray[y:y+h,x:x+w])

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Capture Images",img)

        if cv2.waitKey(1) == ord('q') or count > 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    add_student(student_id,name)