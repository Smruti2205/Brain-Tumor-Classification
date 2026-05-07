import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# -------------------------------------------------------------------
# 🧠 Load Model Safely (fixes the "expects 1 input but got 2" error)
# -------------------------------------------------------------------
model = load_model("final_clean.keras", safe_mode=False, compile=False)

# Class names (in correct order)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Streamlit Page
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Classification (4 Classes)")
st.write("Upload an MRI image to detect tumor type.")


# -------------------------------------------------------------------
# 📤 File Upload
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # ---------------------------------------------------------------
    # 🖼 Correct Input Size for VGG16 → 224 × 224
    # ---------------------------------------------------------------
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------------------------------------------------------
    # 🔮 Prediction
    # ---------------------------------------------------------------
    preds = model.predict(img_array)[0]  # Shape = (4,)
    class_index = np.argmax(preds)
    confidence = preds[class_index] * 100
    predicted_label = CLASS_NAMES[class_index]

    # ---------------------------------------------------------------
    # 📌 Display Result
    # ---------------------------------------------------------------
    st.subheader("📌 Prediction Result")
    st.write(f"**Predicted Class:** `{predicted_label}`")
    st.write(f"**Confidence:** `{confidence:.2f}%`")
    st.progress(int(confidence))

    # ---------------------------------------------------------------
    # 🔎 Show Class Probabilities
    # ---------------------------------------------------------------
    st.markdown("### 🔎 Class Probabilities")
    for i, c in enumerate(CLASS_NAMES):
        st.write(f"**{c}** → {preds[i]*100:.2f}%")
        st.progress(int(preds[i] * 100))
