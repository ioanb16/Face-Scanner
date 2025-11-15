import os
import cv2
from deepface import DeepFace
import sys
import time





script_dir = os.path.dirname(os.path.abspath(__file__))


db_folder_path = os.path.join(script_dir, "Known_Faces")
db_folder_path = os.path.normpath(db_folder_path)


if not os.path.exists(db_folder_path):
    print(f"Error: Database path not found at {db_folder_path}")
    sys.exit()


try:
    old_db = os.path.join(db_folder_path, "representations_sface.pkl")
    if os.path.exists(old_db):
        os.remove(old_db)
        print("Removed old database file. A new one will be created.")
except Exception as e:
    print(f"Could not remove old database file: {e}")




print("Starting webcam...")
video_capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("Error: Cannot open camera at index 0.")
    sys.exit()

last_check_time = time.time()
time_threshold = 1.0 
cached_results = [] 

while True:
    result, video_frame = video_capture.read()
    if result is False:
        break

   
    frame_to_draw = video_frame.copy()
    
    
    current_time = time.time()
    if (current_time - last_check_time) > time_threshold:
        last_check_time = current_time
        
        try:
            
            results = DeepFace.find(img_path=video_frame, 
                                    db_path=db_folder_path, 
                                    model_name='SFace', 
                                    enforce_detection=False)
            
            
            cached_results = results
            
        except Exception as e:
            
            cached_results = []


   
    if cached_results:
        for result_df in cached_results:
            if not result_df.empty:
                for index, row in result_df.iterrows():
                    
                    
                    full_path = row['identity']
                    
                    
                    parent_dir = os.path.dirname(full_path)
                    
                    
                    name = os.path.basename(parent_dir)
                    
                    
                    x = row['source_x']
                    y = row['source_y']
                    w = row['source_w']
                    h = row['source_h']
                    
                    
                    cv2.rectangle(frame_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    
                    cv2.putText(frame_to_draw, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)


    
    cv2.imshow("My Face Recognition Project", frame_to_draw)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()


# model name VGG-Face is slow and SFace is fast but less accurate