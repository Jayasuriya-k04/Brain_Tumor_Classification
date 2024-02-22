import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
import numpy as np

app = Flask(__name__)


model = load_model("Tumour_model(V19).h5")

ref = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'} 

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if request.method == 'POST':
        file = request.files['file']

        if file:
            file_path = 'temp.jpg'
            file.save(file_path)

            img_array = preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_class_name = ref[predicted_class]

            data['success'] = True
            data['prediction'] = {
                'class_index': int(predicted_class),
                'class_name': predicted_class_name,
                'probabilities': predictions.tolist().pop()
            }
            print(data)

    return jsonify(data) 

if __name__ == '__main__':
    app.run(debug=True)