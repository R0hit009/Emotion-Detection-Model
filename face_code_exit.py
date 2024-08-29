
import cv2
import numpy as np
from keras.models import model_from_json
import speech_recognition as sr
import threading
import pyttsx3
import datetime

exit_signal = False
emotion_now = ""


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)  # Speed of speech
    engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand what you said.")
            return ""
        except sr.RequestError as e:
            speak(f"Error connecting to Google's servers: {e}")
            return ""

def main_code():
    global exit_signal
    speak("Hello, I am Jarvis. How can I assist you today?")
    while not exit_signal:
        command = listen()
        
        if "time" in command:
            current_time = datetime.datetime.now().strftime("%H:%M")
            speak(f"The current time is {current_time}")
        elif "exit" in command:
            speak("Goodbye!")
            exit_signal = True
        elif "emotion" or "mood" in command:
            speak(f"it seems you are {emotion_now} now ")
        else:
            speak("I'm sorry, I don't understand that command.")


def video_capture() :
    global emotion_now
    # Play the audio file (on Windows)
    while not exit_signal:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
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
            emotion_now = emotion_dict[maxindex]
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



video_thread = threading.Thread(target=video_capture)
main_thread = threading.Thread(target=main_code)

video_thread.start()
main_thread.start()

# Wait for both threads to finish
video_thread.join()
main_thread.join()