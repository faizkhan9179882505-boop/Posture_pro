import cv2
import mediapipe as mp
import numpy as np
from .pose_utils import calculate_angle, draw_landmarks

mp_pose = mp.solutions.pose

class LungeAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.rep_count = 0
        self.stage = "up"  # "up" or "down"
        self.knee_angles = []
        self.hip_angles = []
        self.torso_angles = []
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
            
            # Get coordinates for left leg
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Get coordinates for right leg
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Get shoulder coordinates for torso angle
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Calculate angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            torso_angle = calculate_angle(left_hip, left_shoulder, 
                                        [left_shoulder[0], left_shoulder[1] - 0.1])  # Vertical reference
            
            # Use the knee with smaller angle (bent knee)
            knee_angle = min(left_knee_angle, right_knee_angle)
            self.knee_angles.append(knee_angle)
            self.hip_angles.append(calculate_angle(left_knee, left_hip, right_hip))
            self.torso_angles.append(torso_angle)
            
            # Lunge counter logic
            if knee_angle < 90 and self.stage == "up":
                self.stage = "down"
                self.rep_count += 1
            elif knee_angle > 160 and self.stage == "down":
                self.stage = "up"
            
            # Provide feedback
            feedback = []
            
            # Knee angle feedback
            if self.stage == "down" and knee_angle > 90:
                feedback.append("Go lower in your lunge")
            elif self.stage == "up" and knee_angle < 160:
                feedback.append("Stand up straight between lunges")
            
            # Torso angle feedback
            if torso_angle < 80:
                feedback.append("Keep your torso upright")
            
            # Knee alignment feedback
            if left_knee[0] < left_ankle[0] or right_knee[0] > right_ankle[0]:
                feedback.append("Keep your knees behind your toes")
            
            if not feedback:
                feedback.append("Good form!")
            
            # Display angles
            cv2.putText(image, f"Knee: {int(knee_angle)}°", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
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
            
            # Display feedback
            for i, fb in enumerate(feedback):
                cv2.putText(image, fb, (10, 100 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw landmarks
            draw_landmarks(image, results.pose_landmarks)
            
        except Exception as e:
            print(f"Error in lunge analysis: {e}")
            pass
            
        return image, feedback[0] if feedback else "Perform lunges"

    def get_overall_feedback(self):
        if not self.knee_angles or not self.hip_angles or not self.torso_angles:
            return "No analysis data available"
            
        avg_knee_angle = sum(self.knee_angles) / len(self.knee_angles)
        avg_hip_angle = sum(self.hip_angles) / len(self.hip_angles)
        avg_torso_angle = sum(self.torso_angles) / len(self.torso_angles)
        
        feedback = []
        feedback.append(f"Total lunges: {self.rep_count}")
        
        if avg_knee_angle > 100:
            feedback.append("Try to go deeper in your lunges")
        else:
            feedback.append("Good depth in your lunges")
            
        if avg_hip_angle < 150:
            feedback.append("Keep your hips square and facing forward")
        else:
            feedback.append("Good hip alignment")
            
        if avg_torso_angle < 80:
            feedback.append("Work on keeping your torso upright")
        else:
            feedback.append("Good torso posture")
            
        return "\n".join(feedback)