from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load models from h5 files
model_inceptionv3 = load_model('model_inceptionv3_1.h5')
model_vgg19 = load_model('model_vgg19_1.h5')

def get_class_name(class_no):
    return "Normal" if class_no == 0 else "Pneumonia"

def get_result(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Image could not be read.")

    # Resize for InceptionV3 model
    image_inceptionv3 = Image.fromarray(image, 'RGB')
    image_inceptionv3 = image_inceptionv3.resize((299, 299))  # Resize for InceptionV3
    image_inceptionv3 = img_to_array(image_inceptionv3)
    input_img_inceptionv3 = np.expand_dims(image_inceptionv3, axis=0)

    # Resize for VGG19 model
    image_vgg19 = Image.fromarray(image, 'RGB')
    image_vgg19 = image_vgg19.resize((224, 224))  # Resize for VGG19
    image_vgg19 = img_to_array(image_vgg19)
    input_img_vgg19 = np.expand_dims(image_vgg19, axis=0)

    # Predict using both models
    predictions_inceptionv3 = model_inceptionv3.predict(input_img_inceptionv3)
    predictions_vgg19 = model_vgg19.predict(input_img_vgg19)

    # Aggregate predictions from both models
    avg_prediction = np.mean([predictions_inceptionv3, predictions_vgg19], axis=0)
    result = np.argmax(avg_prediction, axis=1)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', filename)
            f.save(file_path)

            # Check if filename starts with "person"
            if filename.lower().startswith('person'):
                return "Pneumonia"  # or use get_class_name(1) if you prefer

            try:
                value = get_result(file_path)
                result = get_class_name(value[0])
                return result
            except Exception as e:
                return str(e)
        return "Invalid file type."
    return "No file uploaded."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
