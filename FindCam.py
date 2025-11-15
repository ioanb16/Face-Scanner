import cv2

def find_cameras():
    """Tests camera indices 0 through 4 to see which ones open."""
    
    print("Searching for available cameras...")
    available_cameras = []
    
    # Check indices 0 through 4
    for i in range(5):
        # Try to open the camera
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # The camera opened successfully
            print(f"✅ Camera found at index: {i}")
            available_cameras.append(i)
            # IMPORTANT: Release the camera so it's free for other tests
            cap.release()
        else:
            # The camera did not open
            print(f"❌ No camera at index: {i}")

    if not available_cameras:
        print("\nNo cameras were found. Make sure drivers are installed and the camera is not in use.")
    else:
        print(f"\nAvailable camera indices: {available_cameras}")

if __name__ == "__main__":
    find_cameras()