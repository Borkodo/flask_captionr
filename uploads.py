import os
import glob

from flask import Blueprint, render_template, request, url_for, redirect, send_from_directory, current_app
from .azure_integration import get_caption

uploads_bp = Blueprint('uploads', __name__, template_folder='templates')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_uploads_directory():
    files = glob.glob(os.path.join(current_app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)


@uploads_bp.route('/')
def index():
    clear_uploads_directory()
    return render_template('upload.html')


@uploads_bp.route('/display_image', methods=['POST'])
def display_image():
    if 'file' not in request.files:
        return redirect(url_for('uploads.index'))

    uploaded_file = request.files['file']

    if not uploaded_file or uploaded_file.filename == '':
        return redirect(url_for('uploads.index'))

    if uploaded_file and allowed_file(uploaded_file.filename):
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)
        filename = 'uploads/' + uploaded_file.filename

        caption = get_caption(filepath)
    else:
        return "Invalid file type. Please upload an image."

    return render_template('display_image.html', filename=filename, caption=caption)


@uploads_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
