##### \# AI-Powered Real-Time Sign Language Recognition System

##### 

##### A real-time sign language recognition system that uses computer vision, MediaPipe hand landmark extraction, machine learning classification, sentence construction, and text-to-speech output.

##### 

##### The system recognizes hand signs from a live webcam feed, converts them into text, allows the user to build a sentence, and speaks the sentence using text-to-speech.

##### 

##### \---

##### 

##### \## Project Overview

##### 

##### This project was developed as an AI and computer vision accessibility system. The goal is to support communication by recognizing common hand signs in real time and converting them into readable and spoken output.

##### 

##### The final working version uses MediaPipe hand landmark extraction and a machine learning classifier trained on custom live landmark data.

##### 

##### Earlier experiments used TensorFlow Object Detection API, but the final landmark-based approach provided better real-time stability and accuracy.

##### 

##### \---

##### 

##### \## Supported Signs

##### 

##### The system currently supports 10 signs:

##### 

##### \- hello

##### \- thanks

##### \- yes

##### \- no

##### \- iloveyou

##### \- please

##### \- sorry

##### \- help

##### \- good

##### \- clockit

##### 

##### \---

##### 

##### \## Key Features

##### 

##### \- Real-time webcam-based sign recognition

##### \- MediaPipe hand tracking and landmark extraction

##### \- Custom machine learning classifier

##### \- 10 supported hand signs

##### \- Both-hand support using mirrored landmark augmentation

##### \- Confidence score display

##### \- Top prediction display

##### \- Sentence builder

##### \- Text-to-speech output

##### \- Button-based OpenCV user interface

##### \- Keyboard backup controls

##### 

##### \---

##### 

##### \## Technologies Used

##### 

##### \- Python

##### \- OpenCV

##### \- MediaPipe

##### \- Scikit-learn

##### \- NumPy

##### \- Pandas

##### \- Joblib

##### \- pyttsx3

##### \- TensorFlow Object Detection API

##### \- LabelImg

##### \- Git and GitHub

##### 

##### \---

##### 

##### \## Final System Architecture

##### 

##### Webcam Input  

##### ↓  

##### MediaPipe Hand Detection  

##### ↓  

##### 21 Hand Landmark Extraction  

##### ↓  

##### Landmark Normalization  

##### ↓  

##### Machine Learning Classifier  

##### ↓  

##### Predicted Sign  

##### ↓  

##### Sentence Builder  

##### ↓  

##### Text-to-Speech Output  

##### 

##### \---

##### 

##### \## Machine Learning Approach

##### 

##### The final model uses hand landmark features instead of raw image object detection.

##### 

##### For each detected hand, MediaPipe extracts 21 hand landmarks. Each landmark contains x, y, and z coordinates. This gives 63 numerical features per sample.

##### 

##### The landmark points are normalized relative to the wrist point and scaled to reduce sensitivity to hand position, distance, and size.

##### 

##### A machine learning classifier is trained on the extracted landmark features and used for real-time sign prediction.

##### 

##### The final system uses:

##### 

##### MediaPipe Hands + Landmark Feature Extraction + Scikit-learn Classifier

##### 

##### This approach was selected because it gave better real-time stability than the earlier object detection model.

##### 

##### \---

##### 

##### \## Dataset

##### 

##### The project uses a custom dataset collected using a webcam.

##### 

##### Two dataset approaches were used during development.

##### 

##### \### 1. Image-Based Dataset

##### 

##### This dataset was used for the earlier TensorFlow Object Detection experiment.

##### 

##### It included:

##### 

##### \- webcam image collection

##### \- manual annotation using LabelImg

##### \- Pascal VOC XML annotation files

##### \- train/test splitting

##### \- CSV label conversion

##### \- TFRecord generation

##### \- SSD MobileNet transfer learning

##### 

##### \### 2. Live Landmark Dataset

##### 

##### This dataset is used in the final working version.

##### 

##### Instead of training directly on raw images, hand landmarks were collected from the live webcam using MediaPipe. These landmarks were stored as numerical feature samples and used to train the final sign classifier.

##### 

##### The final model was trained using live landmark samples for these signs:

##### 

##### \- hello

##### \- thanks

##### \- yes

##### \- no

##### \- iloveyou

##### \- please

##### \- sorry

##### \- help

##### \- good

##### \- clockit

##### 

##### Both-hand recognition was improved using mirrored landmark augmentation, allowing the system to better recognize signs from either hand.

##### 

##### \---

##### 

##### \## Training Result

##### 

##### The landmark-based classifier achieved strong offline performance on the collected dataset.

##### 

##### Accuracy: 95.91%

##### 

##### This result was achieved using custom live landmark samples and machine learning classification.

##### 

##### \---

##### 

##### \## Real-Time Application

##### 

##### The final real-time application is implemented in:

##### 

##### src/real\_time\_landmark\_detection.py

##### 

##### The application provides:

##### 

##### \- live webcam recognition

##### \- hand landmark visualization

##### \- detected sign display

##### \- confidence score

##### \- top predictions

##### \- sentence display

##### \- clickable control buttons

##### \- text-to-speech output

##### 

##### \---

##### 

##### \## Application Controls

##### 

##### The OpenCV window includes clickable buttons:

##### 

##### \- ADD WORD: Add the currently detected sign to the sentence

##### \- SPEAK: Speak the current sentence

##### \- CLEAR: Clear the sentence

##### \- BACK: Remove the last word

##### \- QUIT: Close the application

##### 

##### Keyboard backup controls:

##### 

##### \- A = Add word

##### \- S = Speak

##### \- C = Clear

##### \- B = Backspace

##### \- Q = Quit

##### 

##### \---

##### 

##### \## Important Files

##### 

##### \- src/collect\_images.py

##### \- src/split\_dataset.py

##### \- src/xml\_to\_csv.py

##### \- src/generate\_tfrecord.py

##### \- src/collect\_landmark\_data.py

##### \- src/train\_landmark\_classifier.py

##### \- src/train\_live\_landmark\_classifier.py

##### \- src/real\_time\_landmark\_detection.py

##### \- dataset/

##### \- annotations/

##### \- sign\_models/

##### \- landmark\_model/live\_landmarks.csv

##### \- landmark\_model/live\_sign\_classifier.joblib

##### 

##### \---

##### 

##### \## How to Run

##### 

##### \### 1. Clone the Repository

##### 

##### git clone https://github.com/kbrehman/sign-recognition-project.git

##### 

##### cd sign-recognition-project

##### 

##### \### 2. Create and Activate Virtual Environment

##### 

##### python -m venv venv

##### 

##### venv\\Scripts\\activate

##### 

##### \### 3. Install Dependencies

##### 

##### pip install -r requirements.txt

##### 

##### \### 4. Run the Final Real-Time System

##### 

##### python src\\real\_time\_landmark\_detection.py

##### 

##### \---

##### 

##### \## How the Final System Works

##### 

##### 1\. The webcam captures a live video frame.

##### 2\. MediaPipe detects the hand.

##### 3\. The system extracts 21 hand landmarks.

##### 4\. The landmarks are normalized relative to the wrist.

##### 5\. The trained machine learning classifier predicts the sign.

##### 6\. The predicted sign is displayed on the screen.

##### 7\. The user can add the detected sign to a sentence.

##### 8\. The sentence can be spoken using text-to-speech.

##### 

##### \---

##### 

##### \## Earlier TensorFlow Object Detection Experiment

##### 

##### An earlier version of this project used TensorFlow Object Detection API with SSD MobileNet for hand sign object detection.

##### 

##### That approach included:

##### 

##### \- custom image collection

##### \- LabelImg annotation

##### \- Pascal VOC XML annotations

##### \- TFRecord generation

##### \- transfer learning using SSD MobileNet

##### \- real-time bounding box detection

##### \- confidence score display

##### \- sentence builder

##### \- text-to-speech output

##### 

##### However, when the system was expanded to 10 signs, the object detection model became unstable and sometimes detected background regions instead of the hand.

##### 

##### To improve real-time performance, the project was redesigned using MediaPipe hand landmarks and a landmark-based machine learning classifier.

##### 

##### \---

##### 

##### \## Why Landmark-Based Recognition Was Used

##### 

##### The landmark-based approach was selected because:

##### 

##### \- MediaPipe provides reliable hand tracking

##### \- hand landmarks reduce background noise

##### \- the model focuses on hand pose instead of image background

##### \- training is faster

##### \- real-time performance is more stable

##### \- both-hand recognition can be improved using mirror augmentation

##### 

##### This made the final system more suitable for real-time sign recognition.

##### 

##### \---

##### 

##### \## Limitations

##### 

##### \- The current system supports static hand signs only.

##### \- Dynamic gestures are not yet supported.

##### \- Accuracy depends on lighting, hand visibility, and webcam quality.

##### \- The dataset was collected from a limited environment.

##### \- Performance may vary for different users without additional training data.

##### 

##### \---

##### 

##### \## Future Improvements

##### 

##### \- Add more signs

##### \- Add dynamic gesture recognition using LSTM or GRU

##### \- Add Urdu text output

##### \- Add voice selection

##### \- Add user profiles for personalized sign calibration

##### \- Improve dataset diversity with more users

##### \- Add a desktop GUI using Tkinter, PyQt, or Streamlit

##### \- Package the system as a standalone desktop application

##### \- Add multilingual text-to-speech support

##### \- Add real-time sentence correction

##### \- Add support for continuous sign language phrases

##### 

##### \---

##### 

##### \## Project Highlights

##### 

##### This project demonstrates:

##### 

##### \- computer vision

##### \- machine learning

##### \- real-time AI inference

##### \- accessibility-focused design

##### \- human-computer interaction

##### \- custom dataset creation

##### \- landmark feature engineering

##### \- text-to-speech integration

##### \- model iteration and improvement

##### \- Git/GitHub project management

##### 

##### \---



##### \## Demo

##### 

##### A demo video of the real-time sign language recognition system is included in this repository.

##### 

##### \[Watch Demo Video](results/demo/sign\_recognition\_demo.mp4)





##### \## Author

##### 

##### Bushra Rehman

##### 

##### Electrical Engineering / Media Engineering Project

##### 

##### \---

##### 

##### \## Status

##### 

##### Final working version completed and pushed to GitHub.

