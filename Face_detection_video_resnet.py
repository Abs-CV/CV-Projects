import cv2
import dlib

# Load the video
video = cv2.VideoCapture("/Users/personal/Downloads/VID_20210126_134952.mp4")

# Get the video dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter("output.mp4", fourcc, 30.0, (width, height))

# Load a deep learning model for face detection (e.g. dlib's resnet model)
face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the video has ended, break the loop
    if not ret:
        break

    # Detect faces in the frame
    faces = face_detector(frame, 1)

    # For each detected face
    for face in faces:
        # Get the face rectangle
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

        # Crop the face and resize it to the desired size
        

        # Swap the face with a different face (e.g. from a different video or image)
        face_image = cv2.imread("face.jpg")
        face_image = cv2.resize(face_image, (w, h))
        # Paste the new face back onto the original frame
        try:
            frame[y:y+h, x:x+w] = face_image
        except:
            pass
            

    # Write the modified frame to the output video
    output.write(frame)

    # Display the frame
    cv2.imshow("Video", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and output objects
video.release()
output.release()

# Close all windowsq
cv2.destroyAllWindows()


# ## Steps to install dblib

# In[4]:


# get_ipython().system('pip install cmake')


# # In[5]:


# get_ipython().system('pip install dlib')


# In[ ]:




