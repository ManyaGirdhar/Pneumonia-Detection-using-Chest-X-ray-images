import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)


# Double-check the path to the saved model
model_path = 'C:\\Users\\kcg\\Downloads\\converted_keras\\pneumonia.hdf5'

# Verify if the model is loaded correctly
try:
    cnn = tf.keras.models.load_model(model_path)
except Exception as e:
    print("Error loading the model:", e)


# # Load the trained CNN model
# cnn = tf.keras.models.load_model('C:\\Users\\kcg\\Downloads\\converted_keras\\pneumonia.hdf5')

def predict_image(image_path):
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make prediction
    result = cnn.predict(test_image)
    if result[0][0] == 1:
        prediction = 'pneumonia'
    else:
        prediction = 'normal'
    
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/model')
def model():
    return render_template('model.html')

def pdf():
    # Replace 'google_drive_pdf_link' with the actual Google Drive link of your PDF file
    return redirect('https://drive.google.com/file/d/1GzwzGvKb-zjtoj0sgPYf6ZbIvZ16uswP/view?usp=sharing')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the uploaded file to a temporary location
            file_path = 'C:\\Users\\kcg\\Downloads\\converted_keras\\static' + file.filename
            file.save(file_path)
            
            # Make prediction
            prediction = predict_image(file_path)
            
            # Delete the temporary file
            os.remove(file_path)
            
            return jsonify({'prediction': prediction})
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
