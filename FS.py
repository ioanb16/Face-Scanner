import cv2


""" Haar Cascade Classifier for Face Detection """

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


""" Capture Video from Webcam """

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

"""create a face detection function"""


