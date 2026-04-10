import cv2
import mediapipe as mp
import numpy as np
from .pose_utils import calculate_angle, draw_landmarks, get_landmark_coordinates

mp_pose = mp.solutions.pose

class SquatAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.rep_count = 0
        self.stage = "up"  # "up" or "down"
        self.feedback = []
        self.knee_angles = []
        self.hip_angles = []

    def analyze_frame(self, frame):
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
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Calculate angles
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(shoulder, hip, knee)
            
            self.knee_angles.append(knee_angle)
            self.hip_angles.append(hip_angle)
            
            # Squat counter logic
            if knee_angle < 90 and self.stage == "up":
                self.stage = "down"
                self.rep_count += 1
            elif knee_angle > 160 and self.stage == "down":
                self.stage = "up"
            
            # Provide feedback
            feedback = []
            if knee_angle > 170:
                feedback.append("Stand straight")
            elif knee_angle < 90 and self.stage == "down":
                feedback.append("Good depth!")
            else:
                feedback.append("Lower your hips more")
            
            # Draw landmarks and angles
            cv2.putText(image, f"Knee: {int(knee_angle)}°", 
                       tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, f"Hip: {int(hip_angle)}°", 
                       tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Rep counter
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.rep_count), 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Stage
            cv2.putText(image, 'STAGE', (65, 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, self.stage.upper(), 
                       (60, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Feedback
            for i, fb in enumerate(feedback):
                cv2.putText(image, fb, (10, 100 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw landmarks
            draw_landmarks(image, results.pose_landmarks)
            
        except Exception as e:
            print(f"Error in squat analysis: {e}")
            pass
            
        return image, feedback[0] if feedback else "No feedback available"

    def get_overall_feedback(self):
        if not self.knee_angles or not self.hip_angles:
            return "No analysis data available"
            
        avg_knee_angle = sum(self.knee_angles) / len(self.knee_angles)
        avg_hip_angle = sum(self.hip_angles) / len(self.hip_angles)
        
        feedback = []
        feedback.append(f"Total squats: {self.rep_count}")
        
        if avg_knee_angle > 160:
            feedback.append("Try to bend your knees more for better form")
        elif avg_knee_angle < 90:
            feedback.append("Good depth maintained in squats")
            
        if avg_hip_angle < 140:
            feedback.append("Keep your back straight during squats")
            
        return "\n".join(feedback)