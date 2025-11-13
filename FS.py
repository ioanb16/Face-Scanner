import cv2


""" Haar Cascade Classifier for Face Detection """

classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_classifier = cv2.CascadeClassifier(classifier_path)

# Check if the classifier loaded
if face_classifier.empty():
    print(f"Error: Could not load Haar cascade file.")
    exit()


""" Capture Video from Webcam """

# Using (0 + cv2.CAP_DSHOW) is often more stable on Windows apparently
video_capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Check if the camera opened
if not video_capture.isOpened():
    print("Error: Cannot open camera at index 0.")
    exit()


""" create a face detection function """


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


""" process webccam frames """

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # use the function we created earlier to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()