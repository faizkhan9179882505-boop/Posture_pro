import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(image, landmarks, connections=None):
    """Draw pose landmarks on the image."""
    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS
    
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=connections,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )
    return image

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_landmark_coordinates(landmarks, landmark_id, frame_width, frame_height):
    """Get the coordinates of a specific landmark."""
    landmark = landmarks.landmark[landmark_id]
    x = min(int(landmark.x * frame_width), frame_width - 1)
    y = min(int(landmark.y * frame_height), frame_height - 1)
    return (x, y)

def get_landmark_visibility(landmarks, landmark_id):
    """Get the visibility score of a specific landmark."""
    return landmarks.landmark[landmark_id].visibility

def is_landmark_visible(landmarks, landmark_id, threshold=0.5):
    """Check if a landmark is visible in the frame."""
    return get_landmark_visibility(landmarks, landmark_id) > threshold