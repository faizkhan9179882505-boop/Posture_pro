# PosturePro - AI-Powered Exercise Analysis

PosturePro is a cutting-edge web application that uses computer vision and machine learning to analyze exercise form and provide real-time feedback. Perfect your posture and prevent injuries with our advanced pose detection technology.

## Features

- **Real-time Exercise Analysis**: Upload videos of your exercises and get instant feedback
- **Multiple Exercise Support**: Analyze squats, push-ups, planks, and lunges
- **AI-Powered Feedback**: Get detailed form corrections and improvement tips
- **Video Processing**: See your exercises with pose overlays and analysis
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **User Authentication**: Secure login system with session management

## Technology Stack

### Backend
- **Flask**: Python web framework
- **MediaPipe**: Google's pose detection library
- **OpenCV**: Computer vision and video processing
- **NumPy**: Numerical computing
- **MoviePy**: Video editing and processing

### Frontend
- **HTML5 & CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Font Awesome**: Icon library
- **Google Fonts**: Typography
- **Responsive Design**: Mobile-first approach

### Deployment
- **Docker**: Containerization
- **Nginx**: Reverse proxy and static file serving
- **Docker Compose**: Multi-container orchestration

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PosturePro
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Frontend: http://localhost:5000
   - API: http://localhost:5000/api

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Frontend: http://localhost
   - API: http://localhost/api

## Usage

1. **Login**: Use demo credentials:
   - Email: demo@posturepro.com
   - Password: demo123

2. **Select Exercise**: Choose from squats, push-ups, planks, or lunges

3. **Upload Video**: Record or upload a video of your exercise (max 100MB)

4. **Get Analysis**: Receive real-time feedback on your form

5. **View Results**: See processed video with pose overlays and corrections

## API Endpoints

### Authentication
- `POST /login` - User login
- `POST /logout` - User logout
- `GET /check-auth` - Check authentication status

### Exercise Analysis
- `POST /analyze` - Upload and analyze exercise video (requires authentication)

### Static Files
- `GET /results/<filename>` - Serve processed videos
- `GET /` - Serve frontend

## Project Structure

```
PosturePro/
├── app.py                 # Main Flask application
├── requirements.txt         # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── nginx.conf           # Nginx configuration
├── front end/           # Frontend files
│   ├── index.html       # Home page
│   ├── login.html      # Login page
│   ├── exercise.html   # Exercise upload page
│   ├── styles.css      # Stylesheets
│   └── assets/        # Images and static files
├── uploads/            # Temporary upload directory
└── results/           # Processed video storage
```

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'production' for production deployment
- `PYTHONUNBUFFERED`: Set to '1' for proper logging

### Nginx Configuration
- Maximum upload size: 100MB
- Timeout: 300 seconds
- Static file caching: 1 day

## Security Features

- Session-based authentication
- CORS protection
- File upload validation
- Input sanitization
- Secure headers

## Performance Optimizations

- Lazy loading for images
- Optimized video processing
- Efficient pose detection
- Responsive image serving
- Browser caching

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Email: support@posturepro.com
- Documentation: [Link to docs]
- Issues: [Link to GitHub issues]

## Roadmap

- [ ] Real-time camera analysis
- [ ] Exercise progression tracking
- [ ] Social features and sharing
- [ ] Mobile app development
- [ ] Advanced analytics dashboard
- [ ] Integration with fitness wearables
