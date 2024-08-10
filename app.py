from flask import Flask, render_template, request, redirect, url_for
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads_folder():
    # Remove all files in the uploads folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    image_output = ""
    webcam_output = ""

    if request.method == 'POST':
        if 'action' in request.form:
            action = request.form['action']

            if action == 'image_detection':
                # Clear the uploads folder before processing new images
                clear_uploads_folder()

                # Handle image upload and detection
                if 'files[]' not in request.files:
                    return redirect(request.url)

                files = request.files.getlist('files[]')

                # Save the uploaded files
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Define the command to run the TFLite detection on the uploaded images
                command = [
                    "python", 
                    "TFLite_detection_image.py", 
                    "--modeldir=custom_model_lite", 
                    f"--imagedir={app.config['UPLOAD_FOLDER']}", 
                    "--noshow_results", 
                    "--save_results", 
                    f'--folder={app.config["UPLOAD_FOLDER"]}_results'
                ]

                # Run the command and capture only the standard output
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

                # Get the output from stdout
                image_output = result.stdout

            elif action == 'webcam_detection':
                # Handle webcam detection
                command = [
                    "python", 
                    "TFLite_detection_webcam.py", 
                    "--modeldir=custom_model_lite"
                ]

                # Run the command and capture only the standard output
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

                # Get the output from stdout
                webcam_output = result.stdout

    return render_template('index.html', image_output=image_output, webcam_output=webcam_output)

if __name__ == '__main__':
    app.run(debug=True)
