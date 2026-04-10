from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import json
from functools import wraps
import secrets
from werkzeug.utils import secure_filename

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

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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
    
    try:
        # Save the uploaded file
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(filepath)
        
        # Simulate analysis (in real app, this would use MediaPipe)
        import time
        time.sleep(2)  # Simulate processing time
        
        # Generate mock analysis results
        analysis_results = {
            'exercise_type': exercise_type,
            'duration_seconds': 10.5,
            'frame_count': 315,
            'analysis': [
                {
                    'frame': 0,
                    'analysis': {
                        'correct': True,
                        'feedback': ['Good form!'],
                        'angles': {'left_knee': 85, 'right_knee': 83}
                    },
                    'timestamp': 0.0
                },
                {
                    'frame': 105,
                    'analysis': {
                        'correct': False,
                        'feedback': ['Keep your back straighter'],
                        'angles': {'left_knee': 95, 'right_knee': 92}
                    },
                    'timestamp': 3.5
                }
            ],
            'video_url': f'/results/{filename}'
        }
        
        # Save analysis results
        result_json_path = os.path.join(RESULTS_FOLDER, f'analysis_{os.path.splitext(filename)[0]}.json')
        with open(result_json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Move video to results folder
        result_video_path = os.path.join(RESULTS_FOLDER, filename)
        os.rename(filepath, result_video_path)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/about.html')
def serve_about():
    return send_from_directory('front end', 'about.html')

@app.route('/contact.html')
def serve_contact():
    return send_from_directory('front end', 'contact.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('front end', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    app.run(debug=True, port=5002, host='0.0.0.0')
