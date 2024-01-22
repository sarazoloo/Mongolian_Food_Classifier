from fastai.vision.all import *
import streamlit as st
from pathlib import Path
import gdown


st.markdown(""" # Mongolian Food Image Classifier

## There are a few traditional Mongolia dishes, which are:
- Buuz
- Khuushuur
- Niislel Salad
- Tsuivan

This app will identify the uploaded image and will classify it to either one of the four categories.

""")

uploaded_file = st.file_uploader("Upload your image here", type = ['png', 'jpg', 'jpeg'])

# loading the pkl file
model_path = Path('export.pkl')

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1E043bJ--aS0DWZSKnBkHq1Bsgv3OlZAH'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img)

    pred, pred_idx, probs = learn_inf.predict(img)
    st.markdown(f"""## This is an image of a: { pred } """)
    st.markdown(f"""### Confidence level: {round((max(probs.tolist())),2) * 100}%""")


