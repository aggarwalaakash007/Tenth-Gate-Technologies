
# coding: utf-8

# In[ ]:


import socket
import _pickle as cPickle
import struct
import utils
from util import *
from face_new import *
import time
from twilio.rest import Client


# In[ ]:


s = socket.socket()
port = 12001
s.bind(('', port))
s.listen(5)                 # Now wait for client connection.
print("Listening.....")
conn, addr = s.accept()
temp = []
temp.append('Obama')
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,0,255)
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]


# starting lists for calculating modes
emotion_window = []
while True:
    msg = recv_one_message(conn)
    small_frame = cv2.imdecode(msg,1)
    face_id, count1, count, known_face_ids, known_face_encodings, id2text, i, face_location = model(small_frame)
    emotion_window = emotion_detection(small_frame, emotion_classifier, emotion_target_size, emotion_labels, face_location)  
    send_one_message(conn , emotion_window)
    send_one_message(conn , face_id)
    send_one_message(conn , face_location)
    send_one_message(conn, i)
    #print(count)
    #print(count1)
    if count1 != count:
        save_embeddings("data/embd.txt",known_face_ids,known_face_encodings, "data/id2text.txt", id2text)
    #for name in face_id:
        #if name not in temp:
            #accountSid = 'AC2040943c6ef4feae0904b45d49ac9ff0'
            #authToken = 'ce4863c950fa80c423622e4feba5c5eb'
            #twilioClient = Client(accountSid, authToken)
            #myTwilioNumber = '+18283656068'
            #destCellPhone = '+919637827159'
            #myMessage = twilioClient.messages.create(body = "Welcome customer", from_=myTwilioNumber, to=destCellPhone)
            #temp.append(name)

