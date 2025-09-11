import streamlit as st
import cv2
import cvlib as cv
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Colour Detection",
    page_icon="🚗",
    layout="wide"
)

# --- Custom Dark Theme CSS ---
st.markdown("""
<style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    h1 { color: #00A9B7; }
    .stAlert, .st-emotion-cache-1629p8f { background-color: #2D2D2D; border-radius: 10px; padding: 15px; }
    .st-emotion-cache-79elbk { background-color: #2D2D2D; border-radius: 10px; }
    p, .stMarkdown { color: #E0E0E0; }
    .stImage > img { border-radius: 10px; }
    figcaption { color: #AAAAAA !important; }
</style>
""", unsafe_allow_html=True)


def get_dominant_color(image_crop):
    # Using a simple average color for speed
    avg_color_per_row = np.average(image_crop, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color.astype(int)


# --- Main Application ---
st.title("Car Colour and Person Detection in Traffic 🚗")
st.info("This app uses the YOLOv3 model via cvlib and will download model weights on its first run.")

uploaded_file = st.file_uploader("Upload a traffic image to count cars and people.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption='Original Image', use_column_width=True)

    with st.spinner('Detecting objects... This may take a moment on the first run.'):
        # Detect common objects in the image
        bbox, label, conf = cv.detect_common_objects(img_array, confidence=0.25, model='yolov3')

        output_image = np.copy(img_array)
        car_count = 0
        person_count = 0

        # Process each detected object
        for l, b in zip(label, bbox):
            if l == 'car':
                car_count += 1
                (startX, startY, endX, endY) = b

                # Get dominant color of the car
                car_crop = img_array[startY:endY, startX:endX]
                if car_crop.size == 0: continue

                avg_color_bgr = get_dominant_color(car_crop)  # cv2 uses BGR

                # Check if dominant color is blue
                if avg_color_bgr[0] > 100 and avg_color_bgr[1] < 80 and avg_color_bgr[2] < 80:
                    color = (0, 0, 255)  # Red for blue cars
                else:
                    color = (255, 0, 0)  # Blue for other cars

                cv2.rectangle(output_image, (startX, startY), (endX, endY), color, 2)

            elif l == 'person':
                person_count += 1

        st.success("Object detection complete!")
        st.image(output_image, caption='Processed Image', use_column_width=True)

        st.subheader("Detection Summary")
        st.write(f"**Total Cars Detected:** {car_count}")
        st.write(f"**Total People Detected:** {person_count}")
