import cv2
import pickle
import numpy as np
import struct ## new
import face_recognition
import pandas as pd
from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

id2text = {}




def load_embeddings(location_embd, vec_size, location_id2text):
    face_encodings = []
    face_ids = []
    
    with open(location_embd, "r") as fh:
        for line in fh:
            arr = line.split()
            if not arr:
                continue
            word = " ".join(arr[0:-vec_size])
            emb = list(map(float, arr[-vec_size:]))
            face_ids.append(word)
            face_encodings.append(emb)
    with open(location_id2text, "r") as fh:
        for line in fh:
            arr = line.split()
            if not arr:
                continue
            id2text[arr[0]] = " ".join(arr[1:])
    return face_ids, face_encodings, id2text

#function for saving the new customer details
def save_embeddings(location_embd, face_ids, face_encodings, location_id2text, id2text):
    size = len(face_encodings)
    output = ""
    for i in range(size):
        output += str(face_ids[i]) + " " + " ".join(map(str, face_encodings[i])) + "\n"
    with open(location_embd, "w") as f:
        f.write(output)

    output = ""
    for id in id2text:
        output += str(id) + " " + id2text[id] + "\n"
    with open(location_id2text, "w") as f:
        f.write(output)
        
        
def emotion_detection(bgr_image, emotion_classifier, emotion_target_size, emotion_labels, faces):
    emotion_offsets = (20, 40)
    emotion_window = []
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
    return emotion_window

def model(small_frame):
    known_face_ids, known_face_encodings, id2text = load_embeddings("data/embd.txt",128, "data/id2text.txt")
    count = len(id2text)
    count1 = count
    face_encodings = []
    face_ids = []
    image_location = "data/images" 
    rgb_small_frame = small_frame[:, :, ::-1]
    i = 0
    #print(face_locations)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_ids = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = id2text[known_face_ids[first_match_index]]
            i = 1
                
        # If match doesn't found we will append the new embeddings in the known encodings
        # As now they are also our customer
        else:
            #print(face_encoding.length)
            known_face_encodings.append(face_encoding)
            known_face_ids.append(str(count+i))
            id2text[str(count+i)] = str(count+i)
            cv2.imwrite(image_location+"/%d.jpg" % (count+i), rgb_small_frame)
            count1 = count+1
            i = 1
            
        face_ids.append(name)
    return face_ids, count1, count, known_face_ids, known_face_encodings, id2text, i, face_locations


def final_frame(frame , face_id , face_location, emotion_window):
    data = pd.read_csv("customer.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (top, right, bottom, left), name, emotion in zip(face_location, face_id, emotion_window):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        res = name
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        for i in range(data.shape[0]):
            if data.Name[i] == name:
                res = name + ", " + data.Category[i]  
                lv = "Last Visit: " + str(data.Last_visit[i])
                lp = "Last Purchase: " + str(data.Last_purchase[i])
                mn = "Mobile No.: " + str(data.Mobile_No[i])
                cv2.putText(frame, str(lv), (left + 6, bottom + 55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(lp), (left + 6, bottom + 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(mn), (left + 6, bottom + 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        
         # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        
        cv2.putText(frame, str(res), (left + 15, top - 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(emotion), (left-2 , top-2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    return frame    

    
