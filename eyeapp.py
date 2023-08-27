import streamlit as st
import cv2
import mediapipe as mp
import pyautogui

# Initialize Streamlit app
st.title("Eye Controlled Mouse")

button1 = st.button('Click me')
if button1:
    # Initialize MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    # Get screen size
    screen_w, screen_h = pyautogui.size()

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Main Streamlit app loop
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with FaceMesh
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        frame_h, frame_w, _ = frame.shape

        # Process face landmarks
        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Iterate over specific landmarks
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                # Move mouse cursor
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]

            # Process left eye blink
            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)

        # Display the processed frame in Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

# Release resources
    cap.release()
    cv2.destroyAllWindows()
