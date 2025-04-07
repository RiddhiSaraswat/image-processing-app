import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Processing App", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2; font-size: 3rem;'>🎨 Image Processing & Pattern Analysis</h1>
    <p style='text-align: center; font-size: 1.2rem;'>Built with 💙 Streamlit + OpenCV + PIL</p>
    <hr style="border: 1px solid #ddd;">
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if st.sidebar.checkbox("Convert to Grayscale"):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            st.image(gray, caption="Grayscale", use_container_width=True, channels="GRAY")

    # Resize
    if st.sidebar.checkbox("Resize Image"):
        width = st.sidebar.slider("Width", 50, 800, image_np.shape[1])
        height = st.sidebar.slider("Height", 50, 800, image_np.shape[0])
        resized = cv2.resize(image_np, (width, height))
        st.image(resized, caption="Resized Image", use_container_width=True)

    # Rotate
    if st.sidebar.checkbox("Rotate Image"):
        angle = st.sidebar.slider("Angle", -180, 180, 0)
        (h, w) = image_np.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(image_np, M, (w, h))
        st.image(rotated, caption="Rotated Image", use_container_width=True)

    # Gaussian Blur
    if st.sidebar.checkbox("Apply Gaussian Blur"):
        kernel = st.sidebar.slider("Kernel Size (Odd)", 1, 15, 3, step=2)
        blurred = cv2.GaussianBlur(image_np, (kernel, kernel), 0)
        st.image(blurred, caption="Gaussian Blurred Image", use_container_width=True)

    # Median Blur
    if st.sidebar.checkbox("Apply Median Blur"):
        k = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
        median = cv2.medianBlur(image_np, k)
        st.image(median, caption="Median Blurred Image", use_container_width=True)

    # Canny Edge Detection
    if st.sidebar.checkbox("Canny Edge Detection"):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        low = st.sidebar.slider("Min Threshold", 50, 300, 100)
        high = st.sidebar.slider("Max Threshold", 100, 500, 200)
        edges = cv2.Canny(gray, low, high)
        st.image(edges, caption="Canny Edge Detection", use_container_width=True, channels="GRAY")

    # Thresholding
    if st.sidebar.checkbox("Apply Thresholding"):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        threshold_value = st.sidebar.slider("Threshold", 0, 255, 127)
        _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        st.image(thresholded, caption="Thresholded Image", use_container_width=True, channels="GRAY")

    # Contour Detection
    if st.sidebar.checkbox("Detect Contours"):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = image_np.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        st.image(contour_image, caption="Contours", use_container_width=True)

    # Histogram Viewer
    if st.sidebar.checkbox("Show Color Histogram"):
        st.subheader("📊 Color Histogram")
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            histr = cv2.calcHist([image_np], [i], None, [256], [0, 256])
            st.line_chart(histr.flatten())
