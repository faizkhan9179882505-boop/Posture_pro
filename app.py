from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import json
from moviepy.editor import VideoFileClip
import tempfile
import shutil
from functools import wraps
import secrets

app = Flask(__name__, static_folder='front end')
app.secret_key = secrets.token_hex(16)
CORS(app)

# Simple user database (in production, use a proper database)
USERS = {
    'demo@posturepro.com': {
        'password': 'demo123',
        'name': 'Demo User'
    }
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Exercise landmarks and angle calculations
class ExerciseAnalyzer:
    def __init__(self):
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def analyze_squat(self, landmarks):
        """Analyze squat form."""
        # Get the required landmarks
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Check form
        feedback = []
        correct = True
        
        # Check knee angle (should be less than 90 degrees at bottom of squat)
        if left_knee_angle > 100 or right_knee_angle > 100:
            feedback.append("Try to squat deeper for better form.")
            correct = False
        
        # Check knee alignment with toes
        left_knee_alignment = abs(left_knee[0] - left_ankle[0])
        right_knee_alignment = abs(right_knee[0] - right_ankle[0])
        
        if left_knee_alignment > 0.1 or right_knee_alignment > 0.1:
            feedback.append("Keep your knees in line with your toes.")
            correct = False
            
        return {
            'correct': correct,
            'feedback': feedback,
            'angles': {
                'left_knee': left_knee_angle,
                'right_knee': right_knee_angle
            }
        }
    
    def analyze_pushup(self, landmarks):
        """Analyze push-up form."""
        # Get the required landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Calculate body alignment (hip to shoulder line)
        body_alignment = abs(left_hip[1] - left_shoulder[1])
        
        feedback = []
        correct = True
        
        # Check elbow angle (should be close to 90 degrees at bottom)
        if left_elbow_angle > 120 or right_elbow_angle > 120:
            feedback.append("Lower your body more for a proper push-up.")
            correct = False
        elif left_elbow_angle < 60 or right_elbow_angle < 60:
            feedback.append("Don't go too low - maintain control.")
            correct = False
        
        # Check body alignment (should be straight)
        if body_alignment > 0.15:
            feedback.append("Keep your body in a straight line from head to heels.")
            correct = False
        
        # Check hand position
        hand_width = abs(left_wrist[0] - right_wrist[0])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        if hand_width < shoulder_width * 0.8:
            feedback.append("Place your hands slightly wider than shoulder-width apart.")
            correct = False
            
        return {
            'correct': correct,
            'feedback': feedback,
            'angles': {
                'left_elbow': left_elbow_angle,
                'right_elbow': right_elbow_angle
            }
        }
    
    def analyze_plank(self, landmarks):
        """Analyze plank form."""
        # Get the required landmarks
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate body line angles
        avg_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
        avg_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
        avg_ankle = [(left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2]
        
        # Check if body forms straight line
        hip_drop = abs(avg_hip[1] - ((avg_shoulder[1] + avg_ankle[1]) / 2))
        
        feedback = []
        correct = True
        
        # Check hip position
        if hip_drop > 0.1:
            feedback.append("Keep your hips in line with your body - don't let them sag or rise.")
            correct = False
        
        # Check shoulder alignment
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_tilt > 0.05:
            feedback.append("Keep your shoulders level and parallel to the ground.")
            correct = False
            
        return {
            'correct': correct,
            'feedback': feedback,
            'angles': {
                'hip_alignment': hip_drop * 100  # Convert to percentage
            }
        }
    
    def analyze_lunge(self, landmarks):
        """Analyze lunge form."""
        # Get the required landmarks
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        feedback = []
        correct = True
        
        # Determine which leg is forward (lower knee angle)
        front_leg = 'left' if left_knee_angle < right_knee_angle else 'right'
        back_leg = 'right' if front_leg == 'left' else 'left'
        
        # Check front knee angle (should be around 90 degrees)
        front_angle = left_knee_angle if front_leg == 'left' else right_knee_angle
        if front_angle > 110:
            feedback.append("Bend your front knee more - aim for 90 degrees.")
            correct = False
        elif front_angle < 70:
            feedback.append("Don't go too low - maintain good form.")
            correct = False
        
        # Check back knee position
        back_knee_y = left_knee[1] if back_leg == 'left' else right_knee[1]
        back_hip_y = left_hip[1] if back_leg == 'left' else right_hip[1]
        if back_knee_y > back_hip_y + 0.1:
            feedback.append("Lower your back knee closer to the ground.")
            correct = False
        
        # Check knee alignment
        if front_leg == 'left':
            knee_alignment = abs(left_knee[0] - left_ankle[0])
        else:
            knee_alignment = abs(right_knee[0] - right_ankle[0])
            
        if knee_alignment > 0.1:
            feedback.append("Keep your front knee aligned with your ankle.")
            correct = False
            
        return {
            'correct': correct,
            'feedback': feedback,
            'angles': {
                'front_knee': front_angle,
                'back_knee': right_knee_angle if front_leg == 'left' else left_knee_angle
            }
        }

analyzer = ExerciseAnalyzer()

@app.route('/')
def serve_frontend():
    return send_from_directory('front end', 'index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if email in USERS and USERS[email]['password'] == password:
        session['user_id'] = email
        session['user_name'] = USERS[email]['name']
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'email': email,
                'name': USERS[email]['name']
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid email or password'
        }), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    })

@app.route('/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'email': session['user_id'],
                'name': session['user_name']
            }
        })
    else:
        return jsonify({
            'authenticated': False
        }), 401

@app.route('/analyze', methods=['POST'])
@login_required
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    exercise_type = request.form.get('exerciseType', 'squat')
    
    if not video_file:
        return jsonify({'error': 'No video selected'}), 400
    
    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video_file.filename)
    video_file.save(temp_video_path)
    
    try:
        # Process the video
        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        output_path = os.path.join(analyzer.results_dir, f'analyzed_{os.path.basename(video_file.filename)}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        analysis_results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Analyze the pose
                if exercise_type == 'squat':
                    analysis = analyzer.analyze_squat(results.pose_landmarks.landmark)
                elif exercise_type == 'pushup':
                    analysis = analyzer.analyze_pushup(results.pose_landmarks.landmark)
                elif exercise_type == 'plank':
                    analysis = analyzer.analyze_plank(results.pose_landmarks.landmark)
                elif exercise_type == 'lunge':
                    analysis = analyzer.analyze_lunge(results.pose_landmarks.landmark)
                else:
                    analysis = {'correct': False, 'feedback': ['Exercise type not supported'], 'angles': {}}
                
                analysis_results.append({
                    'frame': frame_count,
                    'analysis': analysis,
                    'timestamp': frame_count / fps
                })
                
                # Draw landmarks and analysis on frame
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display feedback
                y_offset = 50
                cv2.putText(frame, f'Exercise: {exercise_type.upper()}', 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if analysis['correct'] else (0, 0, 255), 2)
                
                for i, feedback in enumerate(analysis['feedback']):
                    y_offset += 30
                    cv2.putText(frame, feedback, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Generate analysis summary
        summary = {
            'exercise_type': exercise_type,
            'duration_seconds': frame_count / fps,
            'frame_count': frame_count,
            'analysis': analysis_results,
            'video_url': f'/results/{os.path.basename(output_path)}'
        }
        
        # Save analysis results
        result_json_path = os.path.join(analyzer.results_dir, 
                                      f'analysis_{os.path.splitext(video_file.filename)[0]}.json')
        with open(result_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(analyzer.results_dir, filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True, port=5000)
