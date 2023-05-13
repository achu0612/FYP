# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:42:22 2023

@author: achu6
"""
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for, redirect, make_response, Response, stream_with_context
from tensorflow import keras
import mediapipe as mp

app=Flask(__name__, static_folder='static')

#generate live cam footage and make predictions
def generate_frames():
    camera = cv2.VideoCapture(0)
    model = keras.models.load_model("models/action.h5")
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    sequence = []
    predictions = []
    threshold = 0.5
    ans = ""

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                def mediapipe_detection(image, model):
                    print("mp_detection works")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
                    image.flags.writeable = False                  # Image is no longer writeable
                    results = model.process(image)                 # Make prediction
                    image.flags.writeable = True                   # Image is now writeable 
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
                    return image, results

                def draw_styled_landmarks(image, results):
                    print("draw styled works")
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                            ) 
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            ) 
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            ) 
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            ) 
                    
                def extract_keypoints(results):
                    print("extract works")
                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                    return np.concatenate([pose, face, lh, rh])

                # Actions that we try to detect
                actions = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Good morning", "Hello", "How are you", "I am fine", "Please", "Sorry", "Thank you"])

                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))
                                    
                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            ans = actions[np.argmax(res)]
                 
                
                cv2.rectangle(image, (0,0), (640, 40), (140, 81, 79), -1)
                cv2.putText(image, f"Predicting: {ans}", (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert the frame to a JPEG image
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()

                # Use Flask's stream_with_context to stream the video feed
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n') 

#index homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alphabets')
def alphabets():
    return render_template('alphabets.html')

@app.route('/common-phrases')
def commonphrases():
    return render_template('common-phrases.html')

@app.route('/interpret')
def interpret():
    return render_template('interpret.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/numbers')
def numbers():
    return render_template('numbers.html')

@app.route('/play-video')
def playvideo():
    return render_template('play-video.html')

@app.route('/rendering')
def rendering():
    return render_template('rendering.html', video_url=url_for('video_feed'))

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/selection')
def selection():
    return render_template('selection.html')


if __name__ == '__main__':
    app.run(debug=True, host= '192.168.0.21')

