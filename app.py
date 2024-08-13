import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Load and configure the model
base_model = InceptionV3(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(inputs=base_model.inputs, outputs=output)

# Load InceptionV3 weights (not specific to your model)
model_03.load_weights('model_inceptionv3_1.h5', by_name=True, skip_mismatch=True)

print('Model loaded')

# Define class names
def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

# Predict the class of the given image
def getResult(img):
    # Load and preprocess the image
    image = cv2.imread(img)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")

    print("Original image shape:", image.shape)  # Debugging: Check image shape

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((128, 128))

    # Convert to numpy array
    image = np.array(image)
    print("Processed image shape:", image.shape)  # Debugging: Check processed image shape

    # Preprocess for InceptionV3
    image = preprocess_input(image)
    input_img = np.expand_dims(image, axis=0)
    print("Input image shape for prediction:", input_img.shape)  # Debugging: Check input shape

    # Predict and get class index
    result = model_03.predict(input_img)
    print("Raw prediction result:", result)  # Debugging: Check raw prediction

    result01 = np.argmax(result, axis=1)
    print("Predicted class index:", result01)  # Debugging: Check predicted class index

    return result01

# Define Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f:
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'uploads')
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file_path = os.path.join(upload_folder, secure_filename(f.filename))
            f.save(file_path)

            try:
                value = getResult(file_path)
                result = get_className(value[0])  # get_className expects single classNo
                return result
            except Exception as e:
                return str(e)
    return 'No file uploaded or invalid request.'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
