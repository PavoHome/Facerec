from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import os
import threading

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_DIR = 'known_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load known faces
known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

load_known_faces()

# Initialize video capture
video_capture = cv2.VideoCapture(0)
lock = threading.Lock()

def generate_frames():
    while True:
        with lock:
            ret, frame = video_capture.read()
            if not ret:
                break
            rgb_frame = frame[:, :, ::-1]

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                else:
                    # Save unknown face image
                    unknown_dir = os.path.join('unknown_faces')
                    os.makedirs(unknown_dir, exist_ok=True)
                    face_image = frame[top:bottom, left:right]
                    img_path = os.path.join(unknown_dir, f"unknown_{top}_{left}.jpg")
                    cv2.imwrite(img_path, face_image)

                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        file = request.files['image']
        if file:
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            filepath = os.path.join(person_dir, file.filename)
            file.save(filepath)
            load_known_faces()  # Reload known faces
            return redirect(url_for('index'))
    return render_template('register.html')
@app.route('/label_unknown', methods=['GET', 'POST'])
def label_unknown():
    unknown_dir = 'unknown_faces'
    if request.method == 'POST':
        filename = request.form['filename']
        name = request.form['name']
        source_path = os.path.join(unknown_dir, filename)
        if os.path.exists(source_path):
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            destination_path = os.path.join(person_dir, filename)
            os.rename(source_path, destination_path)
            load_known_faces()  # Reload known faces
            return redirect(url_for('index'))
    else:
        files = os.listdir('unknown_faces') if os.path.exists('unknown_faces') else []
        return render_template('label_unknown.html', files=files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
