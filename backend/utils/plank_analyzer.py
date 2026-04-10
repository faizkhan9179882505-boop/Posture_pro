import cv2
import mediapipe as mp
import numpy as np
from .pose_utils import calculate_angle, draw_landmarks

mp_pose = mp.solutions.pose

class PlankAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.start_time = None
        self.duration = 0
        self.back_angles = []
        self.hip_angles = []
        self.shoulder_angles = []
        self.feedback = []

    def analyze_frame(self, frame):
        if self.start_time is None:
            self.start_time = cv2.getTickCount()
        else:
            self.duration = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for back angle (shoulder-hip-knee)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            # Get coordinates for hip angle (shoulder-hip-ankle)
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Get coordinates for shoulder angle (elbow-shoulder-hip)
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            
            # Calculate angles
            back_angle = calculate_angle(shoulder, hip, knee)
            hip_angle = calculate_angle(shoulder, hip, ankle)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            
            self.back_angles.append(back_angle)
            self.hip_angles.append(hip_angle)
            self.shoulder_angles.append(shoulder_angle)
            
            # Provide feedback
            feedback = []
            
            # Back angle feedback
            if back_angle < 150:
                feedback.append("Keep your back straight!")
            
            # Hip angle feedback
            if hip_angle < 150:
                feedback.append("Hips too high, lower them!")
            elif hip_angle > 190:
                feedback.append("Hips too low, raise them slightly!")
            
            # Shoulder angle feedback
            if shoulder_angle < 60 or shoulder_angle > 100:
                feedback.append("Keep shoulders directly above elbows")
            
            # If no specific issues, provide encouragement
            if not feedback:
                feedback.append("Good form! Keep it up!")
            
            # Display timer
            minutes = int(self.duration // 60)
            seconds = int(self.duration % 60)
            timer_text = f"Time: {minutes:02d}:{seconds:02d}"
            cv2.putText(image, timer_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display angles
            cv2.putText(image, f"Back: {int(back_angle)}°", 
                       tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display feedback
            for i, fb in enumerate(feedback):
                cv2.putText(image, fb, (10, 70 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw landmarks
            draw_landmarks(image, results.pose_landmarks)
            
        except Exception as e:
            print(f"Error in plank analysis: {e}")
            pass
            
        return image, feedback[0] if feedback else "Hold the plank position"

    def get_overall_feedback(self):
        if not self.back_angles or not self.hip_angles or not self.shoulder_angles:
            return "No analysis data available"
            
        avg_back_angle = sum(self.back_angles) / len(self.back_angles)
        avg_hip_angle = sum(self.hip_angles) / len(self.hip_angles)
        avg_shoulder_angle = sum(self.shoulder_angles) / len(self.shoulder_angles)
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        
        feedback = []
        feedback.append(f"Plank duration: {minutes} minutes {seconds} seconds")
        
        if avg_back_angle < 160:
            feedback.append("Try to keep your back straighter during the plank")
        else:
            feedback.append("Good back alignment")
            
        if avg_hip_angle < 160 or avg_hip_angle > 200:
            feedback.append("Work on maintaining a neutral hip position")
        else:
            feedback.append("Good hip position")
            
        if avg_shoulder_angle < 70 or avg_shoulder_angle > 90:
            feedback.append("Keep shoulders directly above your elbows")
        else:
            feedback.append("Good shoulder position")
            
        return "\n".join(feedback)