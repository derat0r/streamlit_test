import streamlit as st
from PIL import Image
import io
from tensorflow.python.keras.models import load_model
import numpy as np


def upload_model():
    model = load_model('model_test1.pt')
    return model


def upload_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознования')
    if uploaded_file is not None:
        image = uploaded_file.getvalue()
        st.image(image)
        return Image.open(io.BytesIO(image)).convert('L')
    else:
        return None


def preprocess_image(img):
    img = img.resize((28, 28))
    img = np.asarray(img)
    img = img / 255
    conv = lambda x: abs(1 - x)
    img = conv(img)
    inp = img.reshape(1, img.T.shape[0], -1)
    return inp


def print_prediction(y):
    pred = np.argmax(y)
    st.write('Распознана цифра: ', pred)


model = upload_model()

st.title('Классификация рукописных цифр')
input_img = upload_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(input_img)
    preds = model.predict(x)
    st.write('**Результаты распознавания**')
    print_prediction(preds)

