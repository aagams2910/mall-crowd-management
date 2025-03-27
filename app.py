from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import threading
import time
import google.generativeai as genai
import base64
import numpy as np
from dotenv import load_dotenv
from people_detector import PeopleDetector

app = Flask(__name__)

# Configure Google Gemini API
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables
output_frame = None
lock = threading.Lock()
detector = None
current_video = 'Floor1.mp4'
current_floor = '1st Floor'
threshold = 25  # Updated threshold to 25
alert_status = False
current_solutions = []
last_frame_analysis_time = 0
frame_analysis_cooldown = 30  # Seconds between frame analyses

# Available floor options with corresponding videos
floor_options = {
    '1st Floor': 'Floor1.mp4',
    '2nd Floor': 'Floor2.mp4',
    '3rd Floor': 'Floor3.mp4'
}

def initialize_detector(video_source='Floor1.mp4', model_path='yolov5su.pt'):
    global detector
    detector = PeopleDetector(model_path=model_path, video_source=video_source, confidence_threshold=0.05)
    return detector

def encode_image_for_gemini(frame):
    """Encode OpenCV frame to base64 for Gemini API"""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None
    
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def analyze_frame_with_gemini(frame, count, floor):
    """Send the frame to Gemini 1.5 Flash for analysis"""
    try:
        encoded_image = encode_image_for_gemini(frame)
        if not encoded_image:
            return get_solutions_from_gemini(count, floor)  # Fallback to text-only if image encoding fails
        
        prompt = f"""
        TASK: Analyze this image of a mall surveillance showing {count} people on the {floor}.
        The crowd threshold is {threshold} people and has been exceeded.
        
        INSTRUCTIONS:
        1. Analyze the crowd distribution, density and movement patterns
        2. Consider potential safety risks or bottlenecks
        3. Generate 4 specific, actionable solutions for managing this exact crowd situation
        
        FORMAT RESPONSE AS A JSON LIST:
        [
          {{"title": "Short solution title", "description": "Brief actionable description (exactly 20-30 words)"}}
        ]
        
        Keep titles under 5 words and descriptions exactly 20-30 words.
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": encoded_image}
        ])
        
        response_text = response.text
        
        # Extract JSON from response (handles potential markdown formatting)
        if "```json" in response_text:
            import json
            import re
            json_str = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_str:
                solutions = json.loads(json_str.group(1))
                # Print solutions in backend console
                print("\n===== VISUAL ANALYSIS SOLUTIONS =====")
                for idx, solution in enumerate(solutions):
                    print(f"{idx+1}. {solution['title']}: {solution['description']}")
                print("====================================\n")
                return solutions
        
        # Try direct JSON parsing
        import json
        try:
            solutions = json.loads(response_text)
            # Print solutions in backend console
            print("\n===== VISUAL ANALYSIS SOLUTIONS =====")
            for idx, solution in enumerate(solutions):
                print(f"{idx+1}. {solution['title']}: {solution['description']}")
            print("====================================\n")
            return solutions
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            print(f"Response text: {response_text}")
            raise Exception(f"Invalid JSON response: {str(json_err)}")
            
    except Exception as e:
        print(f"Error analyzing frame with Gemini: {str(e)}")
        return get_solutions_from_gemini(count, floor)  # Fallback to text-only if image analysis fails

def get_solutions_from_gemini(count, floor):
    try:
        prompt = f"""
        Generate 4 specific crowd management solutions for a shopping mall where {count} people have been detected on the {floor}.
        The crowd threshold is {threshold} people. Format as a JSON list with 'title' and 'description' fields.
        Keep titles under 5 words and descriptions exactly 20-30 words in length.
        Example format: [{{"title": "Solution Title", "description": "Brief description (exactly 20-30 words)"}}]
        """
        
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response (handles potential markdown formatting)
        if "```json" in response_text:
            import json
            import re
            json_str = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_str:
                solutions = json.loads(json_str.group(1))
                # Print solutions in backend console
                print("\n===== TEXT-ONLY SOLUTIONS =====")
                for idx, solution in enumerate(solutions):
                    print(f"{idx+1}. {solution['title']}: {solution['description']}")
                print("==============================\n")
                return solutions
        
        # Try direct JSON parsing
        import json
        try:
            solutions = json.loads(response_text)
            # Print solutions in backend console
            print("\n===== TEXT-ONLY SOLUTIONS =====")
            for idx, solution in enumerate(solutions):
                print(f"{idx+1}. {solution['title']}: {solution['description']}")
            print("==============================\n")
            return solutions
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error: {str(json_err)}")
            print(f"Response text: {response_text}")
            raise Exception(f"Invalid JSON response: {str(json_err)}")
    except Exception as e:
        print(f"Error getting solutions from Gemini: {str(e)}")
        fallback_solutions = [
            {"title": "Redirect Traffic", "description": "Guide visitors to less crowded areas using digital signage and staff at key junctions to maintain smooth flow patterns."},
            {"title": "Staff Deployment", "description": "Position additional personnel at crowd hotspots to manage flow, address concerns and maintain security presence."},
            {"title": "Entry Control", "description": "Temporarily regulate access to affected areas using a one-in-one-out system until density decreases to safe levels."},
            {"title": "Open Alternate Routes", "description": "Unlock emergency or staff pathways to create additional movement options and reduce pressure in main corridors."}
        ]
        # Print fallback solutions
        print("\n===== FALLBACK SOLUTIONS =====")
        for idx, solution in enumerate(fallback_solutions):
            print(f"{idx+1}. {solution['title']}: {solution['description']}")
        print("============================\n")
        return fallback_solutions

def detect_people():
    global output_frame, lock, detector, alert_status, threshold, current_solutions, last_frame_analysis_time
    
    frame_number = 0
    frame_skip = 5
    
    while True:
        if detector is None:
            time.sleep(0.1)
            continue
            
        ret, frame = detector.cap.read()
        if not ret:
            # Restart video when it ends
            detector.cap.release()
            detector.cap = cv2.VideoCapture(current_video)
            continue
            
        if frame_number % frame_skip != 0:
            frame_number += 1
            continue
            
        frame = cv2.resize(frame, (1280, 720))
        results = detector.model(frame)
        human_boxes = [box for box in results[0].boxes if box.cls == 0]
        count = sum(1 for box in human_boxes if box.conf >= detector.confidence_threshold)
        
        for box in human_boxes:
            if box.conf >= detector.confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy().item()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        detector.log_detections(frame_number, count)
        cv2.putText(frame, f"People: {count}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Floor: {current_floor}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Update alert status
        new_alert_status = count > threshold
        
        # If alert status changes to True (crowd exceeds threshold) or 
        # it's been more than the cooldown period since the last analysis
        current_time = time.time()
        if (new_alert_status and 
            (not alert_status or current_time - last_frame_analysis_time > frame_analysis_cooldown)):
            # Use the current frame to analyze with Gemini
            current_solutions = analyze_frame_with_gemini(frame.copy(), count, current_floor)
            last_frame_analysis_time = current_time
            
        alert_status = new_alert_status
        
        if alert_status:
            cv2.putText(frame, "ALERT: CROWD DETECTED!", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        with lock:
            output_frame = frame.copy()
        
        frame_number += 1

def generate():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        time.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html', floor_options=floor_options, current_floor=current_floor, threshold=threshold)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/status')
def status():
    global alert_status, current_solutions
    return jsonify({
        'alert': alert_status,
        'current_floor': current_floor,
        'threshold': threshold,
        'solutions': current_solutions
    })

@app.route('/change_floor', methods=['POST'])
def change_floor():
    global current_floor, current_video, detector
    
    floor = request.json.get('floor')
    if floor in floor_options:
        current_floor = floor
        current_video = floor_options[floor]
        
        # Release current detector and initialize new one
        if detector is not None:
            detector.cap.release()
        
        initialize_detector(video_source=current_video)
        
        return jsonify({'success': True, 'message': f'Changed to {floor}'})
    
    return jsonify({'success': False, 'message': 'Invalid floor option'})

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global threshold
    
    new_threshold = request.json.get('threshold')
    try:
        threshold = int(new_threshold)
        return jsonify({'success': True, 'message': f'Threshold updated to {threshold}'})
    except:
        return jsonify({'success': False, 'message': 'Invalid threshold value'})

if __name__ == '__main__':
    try:
        # Start a thread that will perform people detection
        t = threading.Thread(target=detect_people)
        t.daemon = True
        t.start()
        
        # Initialize detector with default values
        initialize_detector()
        
        port = 49152  # Using a high port number less likely to be blocked
        print(f"Starting server on http://127.0.0.1:{port}")
        app.run(
            host='127.0.0.1',
            port=port,
            debug=True,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        input("Press Enter to exit...")