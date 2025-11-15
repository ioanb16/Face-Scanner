import os
from deepface import DeepFace


script_dir = os.path.dirname(os.path.abspath(__file__))


db_folder_path = os.path.join(script_dir, "Known_Faces")
db_folder_path = os.path.normpath(db_folder_path)



DeepFace.stream(db_path=db_folder_path, 
                model_name='VGG-Face', 
                enable_face_analysis=False,
                time_threshold=1)  


# model name VGG-Face is slow and SFace is fast but less accurate