# sign-recognition-project

advance deep learning model
# AI-Powered Real-Time Sign Language Recognition System



A real-time sign language recognition system built using \*\*Python, OpenCV, TensorFlow Object Detection API, and SSD MobileNet\*\*.  

The system detects custom hand signs from a webcam feed, displays confidence scores, stabilizes predictions using temporal smoothing, builds simple sentences, and converts recognized text to speech.



\---



\## Project Overview



Sign language is an important communication method for people with hearing or speech impairments. This project aims to build a real-time AI-based system that can recognize selected hand signs through a webcam and convert them into readable and spoken text.



The project demonstrates a complete computer vision pipeline:



```text

Webcam

→ Custom image dataset

→ Manual annotation

→ Train/test split

→ TFRecord generation

→ Transfer learning

→ Model export

→ Real-time webcam detection

→ Prediction smoothing

→ Sentence builder

→ Text-to-speech

