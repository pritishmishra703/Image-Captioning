import streamlit as st
import requests
import numpy as np
from PIL import Image
from model import get_caption_model, generate_caption


@st.cache(allow_output_mutation=True)
def get_model():
    return get_caption_model()

caption_model = get_model()

img_url = st.text_input(label='Enter Image URL')

if (img_url != "") or (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    st.image(img)

    img = np.array(img)
    pred_caption = generate_caption(img, caption_model)
    st.write(pred_caption)
