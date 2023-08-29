import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input


model = tf.keras.models.load_model("best_model.h5")  


def main():
    st.title("Plant Disease Detection")
    st.write("Upload plant image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True, width = 10)


        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (256, 256))  # Resize image to match model input shape
        img_array = preprocess_input(img_array)


        pred = np.argmax(model.predict(np.expand_dims(img_array, axis=0)))
        
        ref  = {
			    0: 'Apple___Apple_scab',
			    1: 'Apple___Black_rot',
			    2: 'Apple___Cedar_apple_rust',
			    3: 'Apple___healthy',
			    4: 'Blueberry___healthy',
			    5: 'Cherry_(including_sour)___Powdery_mildew',
			    6: 'Cherry_(including_sour)___healthy',
			    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
			    8: 'Corn_(maize)___Common_rust_',
			    9: 'Corn_(maize)___Northern_Leaf_Blight',
			    10: 'Corn_(maize)___healthy',
			    11: 'Grape___Black_rot',
			    12: 'Grape___Esca_(Black_Measles)',
			    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
			    14: 'Grape___healthy',
			    15: 'Orange___Haunglongbing_(Citrus_greening)',
			    16: 'Peach___Bacterial_spot',
			    17: 'Peach___healthy',
			    18: 'Pepper,_bell___Bacterial_spot',
			    19: 'Pepper,_bell___healthy',
			    20: 'Potato___Early_blight',
			    21: 'Potato___Late_blight',
			    22: 'Potato___healthy',
			    23: 'Raspberry___healthy',
			    24: 'Soybean___healthy',
			    25: 'Squash___Powdery_mildew',
			    26: 'Strawberry___Leaf_scorch',
			    27: 'Strawberry___healthy',
			    28: 'Tomato___Bacterial_spot',
			    29: 'Tomato___Early_blight',
			    30: 'Tomato___Late_blight',
			    31: 'Tomato___Leaf_Mold',
			    32: 'Tomato___Septoria_leaf_spot',
			    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
			    34: 'Tomato___Target_Spot',
			    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
			    36: 'Tomato___Tomato_mosaic_virus',
			    37: 'Tomato___healthy'
              } 

        
        st.write("Predicted Disease:")
        st.write(ref[pred])

if __name__ == "__main__":
    main()

