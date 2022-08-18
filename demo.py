import os.path as osp
import tempfile
from PIL import Image
from inference_backend import load_model, detect_hate_speech

import streamlit as st

image = Image.open('./assets/boun.png')
st.set_page_config(
    page_title="Hate Speech Demo",
    page_icon=image
)

st.title("HDV - Hate Speech Detection")

model, device = load_model(load_from="checkpoints/rulemodel/best_model.pth", is_gpu=0)

input_text = st.text_area("Paste news article", height=400)

if len(input_text) > 0:
    with st.spinner('Analyzing the article...'):
        detected_feats = detect_hate_speech(input_text, model, device)
    if detected_feats["prediction"] == "0":
        st.success('This is not hate speech.', icon="✅")
    elif detected_feats["prediction"] == "1":
        st.error('This is hate speech!', icon="🚨")

