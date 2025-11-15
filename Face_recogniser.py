import os
from deepface import DeepFace

# --- This is the new, improved part ---

# Get the absolute path to the directory where this script is located
# (e.g., C:\Users\barbe\facescanner)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join that path with the name of your database folder
# (e.g., C:\Users\barbe\facescanner\Known_Faces)
db_folder_path = os.path.join(script_dir, "Known_Faces")

# --- End of new part ---

print(f"Starting webcam stream...")
print(f"Using database at: {db_folder_path}") # This will show you the full path it's trying to use

DeepFace.stream(db_path=db_folder_path, 
                model_name='VGG-Face', 
                enable_face_analysis=False)

print("Webcam stream stopped.")