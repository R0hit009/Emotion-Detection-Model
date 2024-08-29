
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import model_from_json


#################################################################

#loading model from disk
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('C:/Users/rohit/OneDrive/Desktop/cnn_ai_projects/Emotion_detection_with_CNN (1)/Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("C:/Users/rohit/OneDrive/Desktop/cnn_ai_projects/Emotion_detection_with_CNN (1)/Emotion_detection_with_CNN-main/model/emotion_model.h5")
print("Loaded model from disk")

##################################################################
#initialize the gui class 
root = tk.Tk()
root.title("Webcam App")

video_capture = cv2.VideoCapture(0)  # Changed camera index to 0 for default camera

canvas = tk.Canvas(root, width=640, height=480)
canvas.grid(row=0, column=0, padx=10, pady=10)
text_box = tk.Text(root, height=20, width=40)
text_box.grid(row=0, column=1, padx=10, pady=10)
button = tk.Button(root, text="Submit") # 1 .... button add function of the cnn
button.grid(row=1, column=0, padx=10, pady=10)



current_image = None
photo = None  # Initialize PhotoImage as None


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('C:/Users/rohit/OneDrive/Desktop/cnn_ai_projects/Emotion_detection_with_CNN (1)/Emotion_detection_with_CNN-main/haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame , scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image=current_image)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # ret, frame = self.video_capture.read()

    # if ret:
    #     self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     self.photo = ImageTk.PhotoImage(image=self.current_image)
    #     self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    # self.window.after(15, self.update_webcam)

root.mainloop()

