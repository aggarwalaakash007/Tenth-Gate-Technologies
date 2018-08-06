
# coding: utf-8

# In[ ]:


import socket
import _pickle as cPickle
import struct
import utils
import cv2
from util import *
from face_new import *
import time


# In[ ]:


cap = cv2.VideoCapture(0)
host = socket.gethostname()
port = 12001           # Reserve a port for your service.
s = socket.socket()
s.connect((host, port))
i =0
while True:
    ret, frame=cap.read()
    print(frame.shape)
    small_frame1 = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    res , small_frame = cv2.imencode('.jpg', small_frame1)
    send_one_message(s, small_frame)
    emotion_window = recv_one_message(s)
    face_id = recv_one_message(s)
    face_location = recv_one_message(s)
    i = recv_one_message(s)
    print(i)
    if i != 0:
        print(face_id)
        frame = final_frame(frame , face_id , face_location , emotion_window)
        
    print(2)
    print(frame.shape)
    cv2.imshow('video' , frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    

