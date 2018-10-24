import boto3
import os
import label_image
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.utils import secure_filename

app = Flask(__name__)
s3 = boto3.resource('s3')
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)
graph_path = os.path.join(os.path.dirname(''), 'tf_files/retrained_graph.pb')
label_path = os.path.join(os.path.dirname(''), 'tf_files/retrained_labels.txt')
output_layer = 'final_result'
input_layer = 'Placeholder'
image_name = 'rg12.jpg'

@app.route('/')
def hello():
    return "Hello World!"


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        return upload_to_s3(file)
    return render_template('upload.html')


def upload_to_s3(file):
    res = ''
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
    file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
    response = label_image.main(graph_path, label_path, output_layer, file_path, input_layer)
    for val in response:
        res = res + ': '.join(str(e) for e in val) + '\n'
    return res
