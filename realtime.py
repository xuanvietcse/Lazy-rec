# Optimize realtime application

import face_recognition
import cv2
import numpy as np
import datetime
import os
import pyrebase

import pyrebase

config = {
    "apiKey":"AIzaSyDpUMYNNoIc_AqJBmt1MMBwhOdrADC6HI8",
    "authDomain":"test-3c6cb.firebaseapp.com",
    "databaseURL":"https://test-3c6cb-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket":"test-3c6cb.appspot.com",
    "serivceAccount":"/home/xuanviet/Downloads/test-3c6cb-firebase-adminsdk-osa8g-b53a0ba248.json"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
xuanviet_image = face_recognition.load_image_file("training/HoangXuanViet_20522149/4d8vzar3.jpg")
xuanviet_face_encoding = face_recognition.face_encodings(xuanviet_image)[0]

known_face_encodings = [
    xuanviet_face_encoding
]
known_face_names = [
    "HoangXuanViet"
]

# Initialize some variables 
Date = datetime.datetime.now()
face_encodings = []
process_this_frame = True
recog_enable = False
attendace_status_table = [-1, 0, 1] # -1: Failed, 0: Success, 1: Idle

def realtime_recog(rgb_small_frame, face_locations, face_encodings, attendance_status):
    attendace_status = attendace_status_table[0]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known faces(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        if name != "Unknown":
            attendace_status = attendace_status_table[1]
        face_names.append(name)
    return face_locations, face_names, attendace_status

while True:
    attendace_status = attendace_status_table[2]
    key = cv2.waitKey(1)
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Only process every other frame of video to save time
    #if process_this_frame:
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0,0), fx=0.25,fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:,:,::-1]

    #TODO:  Fix compute_face_descriptor() on default code
    code = cv2.COLOR_BGR2RGB
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)

    face_locations = []
    face_names = []

    if key & 0xFF == ord('e'):
        recog_enable = True
        print(recog_enable)

    if recog_enable:
        face_locations, face_names, attendace_status = realtime_recog(rgb_small_frame, face_locations, face_encodings, attendace_status)
    # process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    if attendace_status == attendace_status_table[1]:
        print("Checkin Successfully! Information: " + name + " - " + Date.strftime("%H:%M:%S %A %d/%m/%Y "))
        img_name = name + "-" + Date.strftime("%H%M%S-%A-%d%m%Y") + ".jpeg"
        cv2.imwrite(img_name, frame)
        storage.child("AttendanceInformation/" + Date.strftime("%d%m%Y") + "/" + img_name).put(img_name)
        os.remove(img_name)
    elif attendace_status == attendace_status_table[0]:
        print("Checkin Failed!")
    recog_enable = False
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if key & 0xFF == ord('q'):
        break


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()