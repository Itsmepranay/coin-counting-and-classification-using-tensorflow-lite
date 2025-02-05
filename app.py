import os
import glob
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
RESULTS_FOLDER = os.path.join(UPLOAD_FOLDER, '_results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder and results folder exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads_folder():
    """Clear the upload folder by removing all files (but not subdirectories)."""
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Only delete files, not directories (like _results)
        if os.path.isfile(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def clear_results_folder():
    """Clear the results folder by removing all files."""
    for filename in os.listdir(RESULTS_FOLDER):
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.isfile(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_output = ""
    webcam_output = ""
    processed_images = []  # List of processed image filenames

    if request.method == 'POST':
        if 'action' in request.form:
            action = request.form['action']

            if action == 'image_detection':
                # Clear the uploads and results folders before processing new images
                clear_uploads_folder()
                clear_results_folder()

                # Handle image upload and detection
                if 'files[]' not in request.files:
                    return redirect(request.url)

                files = request.files.getlist('files[]')

                # Save the uploaded files
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        print(f"Saved uploaded file: {file_path}")

                # Define the command to run the TFLite detection on the uploaded images
                command = [
                    "python", 
                    "TFLite_detection_image.py", 
                    "--modeldir=custom_model_lite", 
                    f"--imagedir={app.config['UPLOAD_FOLDER']}", 
                    "--noshow_results",
                    "--save_results"
                ]

                # Run the command and capture only the standard output
                print("Running detection command:", " ".join(command))
                subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

                # Look for processed images in RESULTS_FOLDER with jpg or png extensions
                pattern_jpg = glob.glob(os.path.join(RESULTS_FOLDER, "*.jpg"))
                pattern_png = glob.glob(os.path.join(RESULTS_FOLDER, "*.png"))
                pattern_jpeg= glob.glob(os.path.join(RESULTS_FOLDER, "*.jpeg"))
                processed_images = pattern_jpg + pattern_png + pattern_jpeg

                # Extract only the base filenames
                processed_images = [os.path.basename(x) for x in processed_images]

                print("Processed images found:", processed_images)

            elif action == 'webcam_detection':
                # Handle webcam detection
                command = [
                    "python", 
                    "TFLite_detection_webcam.py", 
                    "--modeldir=custom_model_lite"
                ]
                print("Running webcam detection command:", " ".join(command))
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                webcam_output = result.stdout

    return render_template('index.html', processed_images=processed_images, image_output=image_output, webcam_output=webcam_output)

if __name__ == '__main__':
    app.run(debug=True)
