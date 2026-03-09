import tkinter as tk
from database import create_tables
from capture_images import capture
from train_model import train
from recognize_faces import recognize

create_tables()

def capture_ui():
    student_id = entry_id.get()
    name = entry_name.get()
    capture(int(student_id),name)

window = tk.Tk()
window.title("AI Attendance System")
window.geometry("400x350")

title = tk.Label(window,text="Face Recognition Attendance",
                 font=("Arial",16,"bold"))
title.pack(pady=20)

tk.Label(window,text="Student ID").pack()
entry_id = tk.Entry(window)
entry_id.pack()

tk.Label(window,text="Student Name").pack()
entry_name = tk.Entry(window)
entry_name.pack()

tk.Button(window,text="Capture Face",
          command=capture_ui,
          width=20).pack(pady=10)

tk.Button(window,text="Train Model",
          command=train,
          width=20).pack(pady=10)

tk.Button(window,text="Start Attendance",
          command=recognize,
          width=20).pack(pady=10)

window.mainloop()