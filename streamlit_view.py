import streamlit as st 

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from src.facial_key_point.utils.facial_key_point_detection import FacialKeyPointDetection

facial_key_point_detection = FacialKeyPointDetection()

st.markdown('## Facial Key Point Detection')

image = st.file_uploader('Facial Image', ['jpg', 'png', 'jpeg' ], accept_multiple_files=False)

if image is not None:
    image = Image.open(image).convert('RGB')
    st.image(image)
    _, kp = facial_key_point_detection.predict(image)

    draw = ImageDraw.Draw(image)
    point_radius = 2  
    for x, y in zip(kp[0], kp[1]):
        draw.ellipse(
            [(int(x.item()) - point_radius, int(y.item()) - point_radius),
             (int(x.item()) + point_radius, int(y.item()) + point_radius)],
            fill="red"
        )


st.image(image)