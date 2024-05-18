from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report
from time import sleep
import pickle, face_recognition, argparse, cv2, pyrebase, datetime, os
import numpy as np

# Global Define
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
attendance_status_table = [-1, 0, 1] # -1: Failed, 0: Success, 1: IDLE
Date = datetime.datetime.now()

# Firebase Configuration - Put your Firebase Config here!
config = {
    "apiKey":"Your API Key",
    "authDomain":"Your Firebase App domain",
    "databaseURL":"Your database URL",
    "storageBucket":"Your storage bucket",
    "serivceAccount":"Your service account JSON path"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()

# Arguments Parsing
parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument(
    "--train", 
    action="store_true", 
    help="Train on input data"
)
parser.add_argument(
    "--validate", 
    action="store_true", 
    help="Validate trained model"
)
parser.add_argument(
    "--test", 
    action="store_true", 
    help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU) or cnn (GPU)",
)
parser.add_argument(
    "-f", 
    action="store", 
    help="Path to an image with an unknown face"
)
parser.add_argument(
    "--exec", 
    action="store_true", 
    help="Run in realtime"
)
parser.add_argument(
    "--device",
    action="store",
    default='0',
    choices=['0','1'],
    help="Select device you are using? Laptop or Raspberry Pi?"
)
args = parser.parse_args()

# Make sure of file structure
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# Implementation of necessary function
def encode_known_faces(
        model: str = "hog", 
        encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(
            image, model=model
        )
        face_encodings = face_recognition.face_encodings(
            image, face_locations, model='large'
        )

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH
):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations, model='large'
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    name_as_text = ''

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        # Remove print(name, bounding_box)
        _display_face(draw, bounding_box, name)
        name_as_text = name
    del draw
    # pillow_image.show() - disabled for reducing lagging on Pi (debugging purpose)
    print(name_as_text)
    return name_as_text

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.5
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    
def _display_face(draw, bouding_box, name) -> None:
    top, right, bottom, left = bouding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue"
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white"
    )

def validate(model: str="hog") -> None:
    y_true = []
    y_pred = []
    for filepath in Path("validation").glob("*/*"):
        name_true = filepath.parent.name
        y_true.append(name_true)
        if filepath.is_file():
            name_pred = recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )
            y_pred.append(name_pred)
    print(classification_report(y_true=y_true, y_pred=y_pred))

def realtime(
        model: str = "hog",
        device: str = "pc",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    # Initialize some variables
    Button_1, Button_2 = True, True
    camera = -1
    recog_enable = False

    if device == '0':
        camera = input('Choose your camera device? 0: Webcam, 1: ESP-CAM\n')
        if camera == '0':
            # Get a reference to webcam #0 (the default one)
            video_capture = cv2.VideoCapture(0)
        else:
            from urllib.request import urlopen
            ip_address = input('Enter your IP of WebServer: \n')
            url = r'http://' + ip_address + r'/capture'
    elif device == '1':
        camera = input('Choose your camera device? 0: External, 1: ESP-CAM\n')
        if camera == '0':
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
            picam2.start()
        else:
            from urllib.request import urlopen
            ip_address = input('Enter your IP of WebServer: \n')
            url = r'http://' + ip_address + r'/capture'
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(40, GPIO.IN, pull_up_down = GPIO.PUD_UP)
        GPIO.setup(38, GPIO.IN, pull_up_down = GPIO.PUD_UP)
        Button_1 = GPIO.input(40)
        Button_2 = GPIO.input(38)
    else:
        print('Wrong device! Abort!')
        exit(0)

    while True:
        # Default attendance status
        attendance_status = attendance_status_table[2]
        # Event wait
        key = cv2.waitKey(1)

        if device == '0':
            if camera == '0':
                # Grab a single frame of video
                ret, frame = video_capture.read()
            else:
                frame_resp = urlopen(url)
                frame_np = np.asarray(bytearray(frame_resp.read()), dtype="uint8")
                frame = cv2.imdecode(frame_np, -1)
        else:
            if camera == '0':
                frame = picam2.capture_array()
            else:
                frame_resp = urlopen(url)
                frame_np = np.asarray(bytearray(frame_resp.read()), dtype="uint8")
                frame = cv2.imdecode(frame_np, -1)

        # Only process every other frame of video to save time
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

        # Conver the image from BGR color (which OpenCV using) to RGB color (which face_recognition using)
        rgb_small_frame = small_frame[:,:,::-1]

        #TODO: Fix compute_face_descriptor() on sample
        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)

        face_locations = []
        face_names = []

        if key & 0xFF == ord('e') or Button_1 == False:
            recog_enable = True
            sleep(0.3)
        if recog_enable:
            face_locations, face_names, attendance_status = _realtime(rgb_small_frame, face_locations, face_names,encodings_location, attendance_status)

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
            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name,(left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if attendance_status == attendance_status_table[1]:
            print("Checkin Successfully! Information: " + name + " - " + Date.strftime("%H:%M:%S %A %d/%m/%Y "))
            img_name = name + "-" + Date.strftime("%H%M%S-%A-%d%m%Y") + ".jpeg"
            cv2.imwrite(img_name, frame)
            storage.child("AttendanceInformation/" + Date.strftime("%d%m%Y") + "/" + img_name).put(img_name)
            os.remove(img_name)
        elif attendance_status == attendance_status_table[0]:
            print("Checkin Failed!")
        recog_enable = False
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit
        if key & 0xFF == ord('q') or Button_2 == False:
            break
    if device == '0' and camera == '0':
        # Release handle to the webcam
        video_capture.release()
    cv2.destroyAllWindows()

def _realtime(rgb_small_frame, face_locations, face_names, encodings_location, attendance_status):
    attendance_status = attendance_status_table[0]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    for face_encoding in face_encodings:
        # See if the face is match for the known face(s)
        matches = face_recognition.compare_faces(loaded_encodings["encodings"], face_encoding, tolerance=0.5)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(loaded_encodings["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = loaded_encodings["names"][best_match_index]
        if name != "Unknown":
            attendance_status = attendance_status_table[1]
        face_names.append(name)
    return face_locations, face_names, attendance_status

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.exec:
        realtime(model=args.m, device=args.device, encodings_location=DEFAULT_ENCODINGS_PATH)
