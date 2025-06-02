import streamlit as st
from utils.extractor import AadharInfoExtractor, crop_image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def download_model_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)

def load_model_from_gdrive(file_id):
    model_path = "pvc_eaadhar_simplified.h5"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        download_model_from_gdrive(file_id, model_path)
    return load_model(model_path)

@st.cache_resource
def load_model():
    model = load_model_from_gdrive("1AL4_Ge_nSWaXBv6teu5rV1Pf4PnbtDcq")
    #return tf.keras.models.load_model("pvc_eaadhar_simplified.h5", compile=False)
    return model
model = load_model()

st.title("Smart KYC Automation System")
st.write("Upload an Aadhaar card image to extract the details automatically.")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "result_json" not in st.session_state:
    st.session_state.result_json = None

uploaded_file = st.file_uploader("Upload Aadhaar Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

col1, col2 = st.columns([1,1])

with col1:
    if st.button("Extract Info") and st.session_state.uploaded_file is not None:
        with st.spinner("Extracting information from the Aadhaar card..."):
            try:
                img = Image.open(st.session_state.uploaded_file).convert("RGB")
                resized_img = img.resize((224, 224))
                img_array = image.img_to_array(resized_img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = model.predict(img_array)[0][0]
                extractor = AadharInfoExtractor()

                if prediction <= 0.5:
                    cropped = crop_image(st.session_state.uploaded_file)
                    cropped.save("temp_crop.jpg")
                    result_json = extractor.info_extractor("temp_crop.jpg")
                else:
                    result_json = extractor.info_extractor(st.session_state.uploaded_file)

                st.session_state.result_json = result_json
                st.success("Information extracted successfully!")

                st.write(f"**Model confidence for PVC Aadhaar:** {prediction:.2%}")

            except Exception as e:
                st.error(f"Oops! Something went wrong: {e}")

with col2:
    if st.button("Clear"):
        st.session_state.uploaded_file = None
        st.session_state.result_json = None
        st.experimental_rerun = None

if st.session_state.result_json:
    st.subheader("Extracted Information")
    st.json(st.session_state.result_json)

    # Optional: Download button
    st.download_button(
        label="Download Extracted Data as JSON",
        data=st.session_state.result_json,
        file_name="aadhaar_data.json",
        mime="application/json"
    )
