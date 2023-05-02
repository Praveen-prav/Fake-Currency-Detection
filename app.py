from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import base64
import os
# import currencyDetection

app = Flask(__name__)
model = joblib.load('currencyDetector.pkl')

app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        img_file = request.files['image'] # get the file from the form data
        image_data = np.fromstring(img_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        preprocessed_image = preprocessing(image)

        cropped_image = edgeDetection(preprocessed_image)

        features, hm = HoG(cropped_image)

        n = 6000
        if features.shape != (n,):
            # Reshape or pad features to correct shape
            features = np.resize(features, n)

        prediction = model.predict([features])[0]

        if prediction == 0:
            detected = 'real'

        else:
            detected = 'fake'

        return render_template('main.html', test=detected)
    return render_template('main.html')


def preprocessing(img):
    # img = cv2.resize(img, (640, 480))  # Set the desired width and height
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale

    blur = cv2.bilateralFilter(gray, 9, 75, 75) # Apply a bilateral filter to smooth the image while preserving edges
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)  # Apply adaptive thresholding to segment the image into foreground and background

    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the thresholded image

    areas = [cv2.contourArea(c) for c in contours]  # Find the contour with the largest area
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    mask = np.zeros_like(thresh)  # Create a mask for the foreground
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    masked = cv2.bitwise_and(img, img, mask=mask)    # Apply the mask to the original image

    x, y, w, h = cv2.boundingRect(cnt)    # Crop the image
    cropped = masked[y:y+h, x:x+w]

    return cropped

def edgeDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute the gradients in the x and y directions using the Sobel operator
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient and threshold the resulting image
    mag = np.sqrt(dx**2 + dy**2)
    mag_thresh = np.max(mag) * 0.1  # Set the threshold as a fraction of the maximum gradient magnitude
    edges = np.zeros_like(mag)
    edges[mag > mag_thresh] = 255

    return edges


def HoG(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return fd, hog_image


'''
        img_bytes = file.read() # read the file contents as bytes
        img_array = np.frombuffer(img_bytes, dtype=np.uint8) # convert the bytes to a numpy array
        img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR) # read the image using cv2.imread()

        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
        output_size = (500, 250)
        gray = cv2.resize(gray, output_size)
        _, encoded_image = cv2.imencode('.jpg', gray) # encode the grayscale image to JPEG format
        
        output_size = (500, 250)
        img1 = cv2.resize(img, output_size)
        _, encoded_image = cv2.imencode('.jpg', img) # encode the grayscale image to JPEG format
        detect = currencyDetection.detect(image=img)
        if detect == 0:
            test = 'real'
        else:
            test = 'fake'
        # return test, '<img src="data:image/jpeg;base64,{}"/>'.format(base64.b64encode(encoded_image.tobytes()).decode('utf-8'))
'''