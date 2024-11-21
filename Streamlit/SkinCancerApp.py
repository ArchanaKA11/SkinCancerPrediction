# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# #tf.compat.v1.reset_default_graph()
# from keras.models import load_model
# MODEL_PATH = 'your_model.h5'
# model = load_model('cnn_model.h5')
# CLASSES = ['benign', 'malignant']
# st.title("Melanoma Cancer Prediction")
# uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
# if uploaded_image is not None:
#     img = Image.open(uploaded_image)
#     st.image(img, caption="Uploaded Image", use_column_width=True)
#     def preprocess_image(image, target_size):
#         image = image.resize(target_size)
#         image = np.array(image) / 255.0  # Normalize pixel values
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
#         return image
#     processed_image = preprocess_image(img, target_size=(224, 224))  # Adjust based on your model
#     st.write("Making prediction...")
#     predictions = model.predict(processed_image)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     st.write(f"Prediction: {CLASSES[predicted_class]}")

# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
#
# MODEL_PATH = 'cnn_model.h5'  # Ensure this path is correct
# model = load_model(MODEL_PATH)
#
# # Print the model summary to check input shape
# model.summary()
#
# CLASSES = ['benign', 'malignant']
# st.title("Melanoma Cancer Prediction")
#
# uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
# if uploaded_image is not None:
#     img = Image.open(uploaded_image)
#     st.image(img, caption="Uploaded Image", use_column_width=True)
#
#     def preprocess_image(image, target_size):
#         image = image.resize(target_size)
#         image = np.array(image) / 255.0  # Normalize pixel values
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
#         return image
#
#     processed_image = preprocess_image(img, target_size=(224, 224))  # Adjust based on your model
#     st.write("Making prediction...")
#
#     predictions = model.predict(processed_image)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#

# import pandas as pd
# import streamlit as st
# from PIL import Image
# import numpy as np
# import os
# import sys
# from io import BytesIO,StringIO
# from keras.models import load_model
# def main():
#     model=load_model('cnn_model.h5')
#     file=st.file_uploader('Upload file',type=['png','jpg','jpeg'])
#     show_file=st.empty()
#     if not file:
#         show_file.info('Please upload a file'.format(''.join(['png','jpg','jpeg'])))
#         return
#     content=file.getvalue()
#     if isinstance(file,BytesIO):
#         show_file.image(file)
#     else:
#         df=pd.read_csv(file)
#         st.dataframe(df.head(2))
#     file.close()
# main()


# import streamlit as st
# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
#
# def main():
#     st.markdown("""
#     <style>
#                .st-emotion-cache-13ln4jf {
#                 background-color: rgb(0 0 0 / .7);!important;
#                 padding-top:50px !important;
#                 padding-left:40px !important;
#                 padding-right:40px !important;
#
#                 }
#     </style>""", unsafe_allow_html=True)
#
#     st.markdown("<h1 style='text-align:center;'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown("---")
#
#     st.write("""
#     This application is designed to predict the uploaded skin image have Melanoma or not.
#     It uses a deep learning model built with *Keras* to predict whether the uploaded skin image has *benign* or *malignant*.
#     The model was trained on a dataset of images of skin that have benign malignant to recognize their visual features accurately.
#     """)
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])
#
#     if image:
#         image = imread(image)
#         img = resize(image, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         if ind == 0:
#             st.write("benign")
#         else:
#             st.write("malignant")
#
#
# main()


# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
# def main():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #f4f4f4;
#             color: #333;
#         }
#         .header {
#             text-align: center;
#             color: #4CAF50;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#         }
#         .result {
#             text-align: center;
#             font-size: 24px;
#             font-weight: bold;
#             margin-top: 20px;
#         }
#         .image-upload {
#             text-align: center;
#             margin-top: 20px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not. "
#                 "Using a deep learning model built with Keras, it distinguishes between benign and malignant skin conditions.</p>",
#                 unsafe_allow_html=True)
#     st.markdown("---")
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
#
#     if image:
#         img = imread(image)
#         img = resize(img, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         result_text = "Benign" if ind == 0 else "Malignant"
#         st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)
#
# main()




# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
# def main():
#     # Set the background image using CSS
#     st.markdown("""
#     <style>
#         body {
#             background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
#             background-size: cover;
#             background-position: center;
#             color: #333;  /* Default text color */
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent background */
#             padding: 10px;  /* Padding around the text */
#             border-radius: 10px;  /* Rounded corners */
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             background-color: rgba(255, 255, 255, 0.7);
#             border-radius: 10px;
#             color: black;  /* Set text color to black */
#         }
#         .result {
#             text-align: center;
#             font-size: 24px;
#             font-weight: bold;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.7);
#             padding: 10px;
#             border-radius: 10px;
#         }
#         .image-upload {
#             text-align: center;
#             margin-top: 20px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p> ",
#
#                 unsafe_allow_html=True)
#     st.markdown("---")
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])
#
#     if image:
#         img = imread(image)
#         img = resize(img, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         result_text = "Benign" if ind == 0 else "Malignant"
#         st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)
#
# main()


# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
#
# def main():
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home", "Details"])
#
#     if page == "Home":
#         show_home()
#     elif page == "Details":
#         show_details()
#
#
# def show_home():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #FFD700;  /* Yellow background */
#             color: #333;
#         }
#         .header {
#             text-align: center;
#             color: #4CAF50;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>Welcome to Melanoma Cancer Prediction</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p>",
#         unsafe_allow_html=True)
#
#     st.markdown("---")
#     st.file_uploader("Upload an image to predict...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
#
#
# def show_details():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #FFD700;  /* Yellow background */
#             color: #333;
#         }
#         .details {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>Details Page</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<p class='details'>Here you can provide detailed information about the Melanoma detection process, the model used, and any other relevant details.</p>",
#         unsafe_allow_html=True)
#
#     st.markdown("---")
#     st.markdown(
#         "<p class='details'>For more information, please refer to the documentation or consult a medical professional.</p>",
#         unsafe_allow_html=True)
#
#
# main()



# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
# def main():
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home", "Details"])
#
#     if page == "Home":
#         show_home()
#     elif page == "Details":
#         show_details()
#
# def show_home():
#     # Set the background image using CSS
#     st.markdown("""
#     <style>
#         body {
#             background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
#             background-size: cover;
#             background-position: center;
#             color: #333;  /* Default text color */
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent background */
#             padding: 10px;  /* Padding around the text */
#             border-radius: 10px;  /* Rounded corners */
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             background-color: rgba(255, 255, 255, 0.7);
#             border-radius: 10px;
#             color: black;  /* Set text color to black */
#         }
#         .result {
#             text-align: center;
#             font-size: 24px;
#             font-weight: bold;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.7);
#             padding: 10px;
#             border-radius: 10px;
#         }
#         .image-upload {
#             text-align: center;
#             margin-top: 20px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p>", unsafe_allow_html=True)
#     st.markdown("---")
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])
#
#     if image:
#         img = imread(image)
#         img = resize(img, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         result_text = "Benign" if ind == 0 else "Malignant"
#         st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)
#
# def show_details():
#     # Set the background image using CSS
#     st.markdown("""
#     <style>
#         body {
#             background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
#             background-size: cover;
#             background-position: center;
#             color: #333;  /* Default text color */
#         }
#         .details-header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .details {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='details-header'>Details Page</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='details'>This section provides detailed information about Melanoma detection and the model used.</p>", unsafe_allow_html=True)
#     st.markdown("<p class='details'>You can learn more about the symptoms, treatment options, and the importance of early detection.</p>", unsafe_allow_html=True)
#
#     # Additional details or links can go here
#     st.markdown("---")
#     st.markdown("<p class='details'>For more information, please consult medical professionals or refer to reputable sources.</p>", unsafe_allow_html=True)
#
# if __name__ == "__main__":
#     main()



# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
# def main():
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home","Prediction", "Details"])
#
#     if page == "Prediction":
#         show_prediction()
#     elif page == "Details":
#         show_details()
#
# def show_prediction():
#     # Set the background image using CSS
#     st.markdown("""
#     <style>
#         body {
#             background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
#             background-size: cover;
#             background-position: center;
#             color: #333;  /* Default text color */
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.8);
#             padding: 10px;
#             border-radius: 10px;
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             background-color: rgba(255, 255, 255, 0.7);
#             border-radius: 10px;
#             color: black;  /* Set text color to black */
#         }
#         .result {
#             text-align: center;
#             font-size: 24px;
#             font-weight: bold;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.7);
#             padding: 10px;
#             border-radius: 10px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p>", unsafe_allow_html=True)
#     st.markdown("---")
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])
#
#     if image:
#         img = imread(image)
#         img = resize(img, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         result_text = "Benign" if ind == 0 else "Malignant"
#         st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)
#
# def show_details():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #FFD700;  /* Yellow background */
#             color: #333;
#         }
#         .header {
#             text-align: center;
#             color: #4CAF50;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .details {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>Details</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='details'>This page provides detailed information about Melanoma detection and the model used for predictions.</p>", unsafe_allow_html=True)
#     st.markdown("---")
#     st.markdown("<p class='details'>You can find information on how the model works, training data, and its accuracy metrics.</p>", unsafe_allow_html=True)
#
# main()


# import streamlit as st
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
#
# # Load the model
# model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")
#
#
# def main():
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.selectbox("Select a page:", ["Home", "Prediction", "Details"])
#
#     if page == "Prediction":
#         show_prediction()
#     elif page == "Details":
#         show_details()
#
#
# def show_prediction():
#     # Set the background image using CSS
#     st.markdown("""
#     <style>
#         body {
#             background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
#             background-size: cover;
#             background-position: center;
#             color: #333;  /* Default text color */
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.8);
#             padding: 10px;
#             border-radius: 10px;
#         }
#         .description {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             background-color: rgba(255, 255, 255, 0.7);
#             border-radius: 10px;
#             color: black;  /* Set text color to black */
#         }
#         .result {
#             text-align: center;
#             font-size: 24px;
#             font-weight: bold;
#             margin-top: 20px;
#             background-color: rgba(255, 255, 255, 0.7);
#             padding: 10px;
#             border-radius: 10px;
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>MELANOMA CANCER PREDICTION</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p>",
#         unsafe_allow_html=True)
#     st.markdown("---")
#
#     image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])
#
#     if image:
#         img = imread(image)
#         img = resize(img, (150, 150, 1))
#         img = img.reshape(1, 150, 150, 1)
#         y_new = model.predict(img)
#         ind = y_new.argmax()
#
#         result_text = "Benign" if ind == 0 else "Malignant"
#         st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)
#
#
# def show_details():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #FFD700;  /* Yellow background */
#             color: #333;
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .details {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>Details</h1>", unsafe_allow_html=True)
#
#     st.markdown(
#         "<p class='details'><a href='https://colab.research.google.com/drive/1o_po-xT99wIbs730otU6Ofu1DoVuHxHs'>Google Colab</a></p>",
#         unsafe_allow_html=True)
#     st.markdown(
#         "<p class='details'><a href='https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images'>Dataset</a></p>",
#         unsafe_allow_html=True)
#
#
# main()


import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r"C:\Users\DEE\pythonProject3\SkinCancer\cnn_model.h5")


def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Home", "Prediction", "Details"])

    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()
    elif page == "Details":
        show_details()


def show_home():
    # Set the background image and styles using CSS
    st.markdown("""
    <style>
        body {
            background-color: #FFD700;  /* Yellow background */
            color: #333;
        }
        .header {
            text-align: center;
            color: #FFD700;
            font-size: 36px;
            margin-top: 20px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
            padding: 0 20%;
            color: white;
        }
        .flower {
            text-align: center;
            animation: flower-animation 2s infinite;
        }
        @keyframes flower-animation {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='header'>Welcome to Melanoma Cancer Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='description'>Melanoma is a kind of skin cancer that starts in the melanocytes. "
        "Melanocytes are cells that make the pigment that gives skin its color. The pigment is called melanin. "
        "This illustration shows melanoma cells extending from the surface of the skin into the deeper skin layers."
        "This application helps in predicting the presence of Melanoma in skin images using deep learning techniques.</p>",
        unsafe_allow_html=True)

    # Animated flower
    st.markdown(
        "<div class='flower'><img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXDA5pFavgKZujvq7O5ec3czN3uBQdTnNQBw&s' width='100' /></div>",
        unsafe_allow_html=True)


def show_prediction():
    # Set the background image using CSS
    st.markdown("""
    <style>
        body {
            background-image: url('https://your-image-url.com/image.jpg');  /* Replace with your image URL */
            background-size: cover;
            background-position: center;
            color: #333;  /* Default text color */
        }
        .header {
            text-align: center;
            color: #FFD700;
            font-size: 36px;
            margin-top: 20px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
        }
        .description {
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
            padding: 0 20%;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            color: black;  /* Set text color to black */
        }
        .result {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            background-color: rgba(255, 255, 255, 0.7);
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='header'>Make Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='description'>This application predicts whether the uploaded skin image indicates Melanoma or not.</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    image = st.file_uploader("Choose an image to predict...", type=["jpg", "jpeg", "png"])

    if image:
        img = imread(image)
        img = resize(img, (150, 150, 1))
        img = img.reshape(1, 150, 150, 1)
        y_new = model.predict(img)
        ind = y_new.argmax()

        result_text = "Benign" if ind == 0 else "Malignant"
        st.markdown(f"<div class='result'>{result_text}</div>", unsafe_allow_html=True)


# def show_details():
#     st.markdown("""
#     <style>
#         body {
#             background-color: #FFD700;  /* Yellow background */
#             color: #333;
#         }
#         .header {
#             text-align: center;
#             color: #FFD700;
#             font-size: 36px;
#             margin-top: 20px;
#         }
#         .details {
#             text-align: center;
#             font-size: 18px;
#             margin-bottom: 40px;
#             padding: 0 20%;
#             color: black;  /* Set text color to black */
#         }
#     </style>
#     """, unsafe_allow_html=True)
#
#     st.markdown("<h1 class='header'>Details</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<p class='details'><a href='https://colab.research.google.com/drive/1o_po-xT99wIbs730otU6Ofu1DoVuHxHs'>Google Colab</a></p>",
#         unsafe_allow_html=True)
#     st.markdown(
#         "<p class='details'><a href='https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images'>Dataset</a></p>",
#         unsafe_allow_html=True)
def show_details():
    st.markdown("<h2 style='text-align:center;'>Model Details</h2>", unsafe_allow_html=True)
    # st.write("""
    # The model is a convolutional neural network (CNN) trained on a dataset of images to classify individuals as either drowsy or natural.
    # It utilizes various layers to extract features and make accurate predictions.
    # - *Input Layer*: Takes images resized to 150x150 pixels.
    # - *Convolutional Layers*: Extract features from images.
    # - *Pooling Layers*: Reduce dimensionality.
    # - *Dense Layers*: Final classification.
    #
    # Ensure to upload clear images for better prediction results.
    # """)

    st.markdown("<h3>References</h3>", unsafe_allow_html=True)
    st.markdown("[Link to google colab](https://colab.research.google.com/drive/1-H_b4hEH2EgRAfYZUDc8q8vEwuW4CZiT)", unsafe_allow_html=True)
    st.markdown("[Link to Dataset](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset)", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
