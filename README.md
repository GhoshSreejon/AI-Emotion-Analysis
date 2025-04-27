# AI-Based Emotion & Behavior Analyzer

## Overview

The **Emotion & Behavior Analyzer** is a Python-based application that utilizes **DeepFace**, **MediaPipe**, and **Streamlit** to analyze human emotions and behaviors from video input (either a live camera feed or uploaded video). The application detects emotions, hand gestures, eye direction, and head movements to provide a behavioral analysis report.

## Features

- **Real-time Emotion Detection** using DeepFace
- **Hand Gesture Analysis** (Tense vs. Relaxed) using MediaPipe Hands
- **Eye Movement Tracking** (Left, Right, Center) using MediaPipe Face Mesh
- **Head Movement Detection** (Up, Down, Still)
- **Behavioral Analysis Report** generated based on collected data
- **Multiple Modes:**
  - **Detective Mode**: General emotion and behavior detection
  - **Student Behavior Mode**: Focuses on tracking attentiveness
  - **Interview Mode**: Analyzes emotions and body language during interviews
- **AI-Powered Behavior Insights** using Google Gemini API

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip package manager
- OpenCV
- NumPy
- Streamlit
- DeepFace
- MediaPipe

### Install Dependencies

```bash
pip install opencv-python numpy streamlit deepface mediapipe google-generativeai
```

## Usage

### Run the Application

```bash
streamlit run emoapp.py
```

### Select a Mode

- Choose from **Detective**, **Student Behavior**, or **Interview** modes.
- Select the input source (**Live Camera** or **Upload a Video**).
- Click **Start Analysis** to begin the detection.
- Click **Stop Analysis** to generate the report.

## Technologies Used

- **Python**: Primary programming language
- **OpenCV**: Image processing
- **DeepFace**: Emotion detection
- **MediaPipe**: Hand and face analysis
- **Streamlit**: Web-based UI
- **Google Gemini API**: AI-powered behavior insights

## File Structure

```
📂 Emotion-Behavior-Analyzer
│── emoapp.py                 # Main Streamlit app
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

