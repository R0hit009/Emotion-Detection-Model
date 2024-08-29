import streamlit as st
import cv2
import numpy as np

st.title("Webcam Live Feed")
st.write("This application captures images from your webcam.")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

camera.release()
