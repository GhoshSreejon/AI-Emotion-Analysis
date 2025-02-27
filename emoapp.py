import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
import mediapipe as mp
import datetime
import json
from google import genai

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Global variables for behavior analysis
if "behavior_report" not in st.session_state:
    st.session_state.behavior_report = []
if "emotion_totals" not in st.session_state:
    st.session_state.emotion_totals = {"happy": 0, "sad": 0, "angry": 0, "surprise": 0, "fear": 0, "disgust": 0, "neutral": 0}
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "hand_gesture_count" not in st.session_state:
    st.session_state.hand_gesture_count = {"tense": 0, "relaxed": 0}
if "eye_direction_count" not in st.session_state:
    st.session_state.eye_direction_count = {"left": 0, "right": 0, "center": 0}
if "head_movement_count" not in st.session_state:
    st.session_state.head_movement_count = {"up": 0, "down": 0, "still": 0}

# State to control the camera
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


# Function to log data to the report
def log_to_report(mode, analysis):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.behavior_report.append(f"{timestamp} - {mode}: {analysis}")


# Function to analyze emotion using DeepFace
def analyze_emotion(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    emotion_scores = result[0]['emotion']
    return emotion_scores


# Function to analyze hand gestures
def analyze_hands(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Example: Check if the hand is tense (fingers closed) or relaxed (fingers open)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)

            if distance < 0.1:
                st.session_state.hand_gesture_count["tense"] += 1
            else:
                st.session_state.hand_gesture_count["relaxed"] += 1

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


# Function to analyze eye direction and head movement
def analyze_face(frame, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Example: Check eye direction (left, right, center)
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            if left_eye.x < 0.4:
                st.session_state.eye_direction_count["left"] += 1
            elif right_eye.x > 0.6:
                st.session_state.eye_direction_count["right"] += 1
            else:
                st.session_state.eye_direction_count["center"] += 1

            # Example: Check head movement (up, down, still)
            nose_tip = face_landmarks.landmark[4]  # Nose tip landmark
            if nose_tip.y < 0.4:
                st.session_state.head_movement_count["up"] += 1
            elif nose_tip.y > 0.6:
                st.session_state.head_movement_count["down"] += 1
            else:
                st.session_state.head_movement_count["still"] += 1

            # Draw face landmarks on the frame
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)


# Function to start analysis based on the selected mode and input source
def start_analysis(mode, input_source):
    if input_source == "camera":
        cap = cv2.VideoCapture(0)
    else:
        file_path = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if file_path is None:
            st.warning("Please upload a video file.")
            return
        cap = cv2.VideoCapture(file_path.name)

    stframe = st.empty()  # Streamlit placeholder for the video frame

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
            mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:

        while st.session_state.camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze the emotion using DeepFace
            emotion_scores = analyze_emotion(frame)

            # Accumulate the emotion percentages over time (for each frame)
            for emotion, score in emotion_scores.items():
                st.session_state.emotion_totals[emotion] += score
            st.session_state.frame_count += 1

            # Analyze hand gestures
            analyze_hands(frame, hands)

            # Analyze eye direction and head movement
            analyze_face(frame, face_mesh)

            # Find the dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            dominant_emotion_score = emotion_scores[dominant_emotion]

            # Display the dominant emotion and its percentage
            emotion_text = f"{dominant_emotion}: {dominant_emotion_score:.2f}%"
            frame = cv2.putText(frame, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the percentages for all emotions
            y_offset = 100
            for emotion, score in emotion_scores.items():
                frame = cv2.putText(frame, f"{emotion}: {score:.2f}%", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0), 2)
                y_offset += 30

            # Mode-based behavior analysis
            if mode == "Detective":
                log_to_report(mode, f"Detecting: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
            elif mode == "Student Behavior":
                log_to_report(mode, f"Tracking: {dominant_emotion} ({dominant_emotion_score:.2f}%)")
            elif mode == "Interview":
                log_to_report(mode, f"Analyzing: {dominant_emotion} ({dominant_emotion_score:.2f}%)")

            # Display the frame with the predicted emotion and percentages
            stframe.image(frame, channels="BGR", use_container_width=True)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the report based on accumulated emotions and body language
    generate_report(mode)


# Function to generate the report and display it in Streamlit
def generate_report(mode):
    if st.session_state.frame_count == 0:  # Avoid division by zero
        st.session_state.frame_count = 1

    # Calculate the average percentage for each emotion
    average_emotion_scores = {emotion: score / st.session_state.frame_count for emotion, score in st.session_state.emotion_totals.items()}

    # Write the report content
    st.subheader("Emotion & Behavior Analysis Report")
    st.write("=" * 40)

    # Write the emotion percentages
    st.write(f"Analysis Mode: {mode}\n")
    for emotion, score in average_emotion_scores.items():
        st.write(f"{emotion.capitalize()}: {score:.2f}%")

    # Write the body language analysis
    st.write("\nBody Language Analysis:")
    st.write(f"Tense Hand Gestures: {st.session_state.hand_gesture_count['tense']}")
    st.write(f"Relaxed Hand Gestures: {st.session_state.hand_gesture_count['relaxed']}")
    st.write(f"Eye Direction (Left): {st.session_state.eye_direction_count['left']}")
    st.write(f"Eye Direction (Right): {st.session_state.eye_direction_count['right']}")
    st.write(f"Eye Direction (Center): {st.session_state.eye_direction_count['center']}")
    st.write(f"Head Movement (Up): {st.session_state.head_movement_count['up']}")
    st.write(f"Head Movement (Down): {st.session_state.head_movement_count['down']}")
    st.write(f"Head Movement (Still): {st.session_state.head_movement_count['still']}")

    analysis = ""
    for emotion, score in average_emotion_scores.items():
        analysis += f"{emotion.capitalize()}: {score:.2f}%\n"

    analysis += f"Tense Hand Gestures: {st.session_state.hand_gesture_count['tense']}\nRelaxed Hand Gestures: {st.session_state.hand_gesture_count['relaxed']}\nEye Direction (Left): {st.session_state.eye_direction_count['left']}\nEye Direction (Right): {st.session_state.eye_direction_count['right']}\nEye Direction (Center): {st.session_state.eye_direction_count['center']}\nHead Movement (Up): {st.session_state.head_movement_count['up']}\nHead Movement (Down): {st.session_state.head_movement_count['down']}\nHead Movement (Still): {st.session_state.head_movement_count['still']}\n"

    # Gemini API call
    client = genai.Client(api_key="AIzaSyAFsZjer2IRBvB83I7FrPDVVMK484JLZsE")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"""Based on this Emotion & Behavior Analysis for a {mode}: {analysis}, 
    use this JSON format to structure the output:

    {{
        "behavior": "describe overall behavior based on the analysis",
        "action": "suggest appropriate action"
    }}
    """
    )

    ans = response.text
    ans = json.loads(ans[8:-4])
    st.write("\nBehavioral Analysis:")
    st.write(f"Behavior: {ans['behavior']}")
    st.write(f"Suggested action: {ans['action']}")

    st.write("\nAnalysis Completed.")


# Streamlit UI
st.title("Emotion & Behavior Analyzer")
st.sidebar.header("Settings")

# Mode selection
mode = st.sidebar.selectbox("Select Mode", ["Detective", "Student Behavior", "Interview"])

# Input source selection
input_source = st.sidebar.radio("Select Input Source", ["camera", "video"])

# Start analysis button
if st.sidebar.button("Start Analysis"):
    st.session_state.camera_running = True
    start_analysis(mode, input_source)

# Stop analysis button
if st.sidebar.button("Stop Analysis"):
    st.session_state.camera_running = False
    st.success("Analysis stopped. Generating report...")
    generate_report(mode)
