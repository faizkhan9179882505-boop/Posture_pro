import cv2
import mediapipe as mp
import numpy as np
from .pose_utils import calculate_angle, draw_landmarks

mp_pose = mp.solutions.pose

class PushupAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.rep_count = 0
        self.stage = "up"  # "up" or "down"
        self.elbow_angles = []
        self.shoulder_angles = []
        self.feedback = []

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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            
            self.elbow_angles.append(elbow_angle)
            self.shoulder_angles.append(shoulder_angle)
            
            # Pushup counter logic
            if elbow_angle > 160 and self.stage == "down":
                self.stage = "up"
                self.rep_count += 1
            elif elbow_angle < 90 and self.stage == "up":
                self.stage = "down"
            
            # Provide feedback
            feedback = []
            if self.stage == "down" and elbow_angle > 90:
                feedback.append("Go lower!")
            elif self.stage == "up" and elbow_angle < 160:
                feedback.append("Push all the way up!")
            else:
                feedback.append("Good form!")
            
            # Draw angles
            cv2.putText(image, f"Elbow: {int(elbow_angle)}°", 
                       tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
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
            print(f"Error in pushup analysis: {e}")
            pass
            
        return image, feedback[0] if feedback else "No feedback available"

    def get_overall_feedback(self):
        if not self.elbow_angles or not self.shoulder_angles:
            return "No analysis data available"
            
        avg_elbow_angle = sum(self.elbow_angles) / len(self.elbow_angles)
        avg_shoulder_angle = sum(self.shoulder_angles) / len(self.shoulder_angles)
        
        feedback = []
        feedback.append(f"Total pushups: {self.rep_count}")
        
        if avg_elbow_angle > 150:
            feedback.append("Try to go lower in your pushups")
        elif avg_elbow_angle < 90:
            feedback.append("Good depth in your pushups")
            
        if avg_shoulder_angle < 160:
            feedback.append("Keep your body straight during pushups")
        else:
            feedback.append("Good body alignment")
            
        return "\n".join(feedback)