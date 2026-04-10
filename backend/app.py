from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import mediapipe as mp
from datetime import datetime
import json
import uuid
from utils.pose_utils import draw_landmarks, calculate_angle
from utils.squat_analyzer import SquatAnalyzer
from utils.pushup_analyzer import PushupAnalyzer
from utils.plank_analyzer import PlankAnalyzer
from utils.lunge_analyzer import LungeAnalyzer

# Set up Flask with static files from both frontend and backend
app = Flask(__name__, 
            static_folder='../front end', 
            template_folder='../front end',
            static_url_path='')

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def index():
    return send_file('../front end/index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.endswith(('.js', '.css', '.jpg', '.png', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot')):
        return send_from_directory('../front end', path)
    # For any other route, serve index.html to support client-side routing
    return send_file('../front end/index.html')

# Add your analysis routes here
@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    exercise_type = request.form.get('exercise', 'squat').lower()
    
    # Validate exercise type
    valid_exercises = ['squat', 'pushup', 'plank', 'lunge']
    if exercise_type not in valid_exercises:
        return jsonify({'error': f'Invalid exercise type. Must be one of: {valid_exercises}'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{exercise_type}_{timestamp}_{unique_id}.mp4"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialize the appropriate analyzer based on exercise type
            if exercise_type == 'squat':
                analyzer = SquatAnalyzer()
            elif exercise_type == 'pushup':
                analyzer = PushupAnalyzer()
            elif exercise_type == 'plank':
                analyzer = PlankAnalyzer()
            elif exercise_type == 'lunge':
                analyzer = LungeAnalyzer()
            
            # Process the video
            result = analyzer.analyze(filepath)
            
            # Save processed video if available
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            if hasattr(analyzer, 'save_processed_video') and callable(analyzer.save_processed_video):
                analyzer.save_processed_video(processed_filepath)
            
            # Return analysis results
            return jsonify({
                'status': 'success',
                'message': 'Analysis completed successfully',
                'exercise': exercise_type,
                'results': result,
                'original_video': filename,
                'processed_video': processed_filename if os.path.exists(processed_filepath) else None
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)