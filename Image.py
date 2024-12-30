import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
import streamlit as st

# Load the dataset
file_path = "Color_Combinations.csv"  # Path to your CSV file
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip().str.lower()

# Dataset columns (e.g., color 1 and color 2 names)
color1_col = 'color 1 name'
color2_col = 'color 2 name'

# Load Haar cascades for detecting the upper and lower body
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
lower_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')

# Function to find the dominant color in a region
def get_dominant_color(image, region):
    """Extract dominant color from the specified region of the image."""
    x, y, w, h = region
    cropped_image = image[y:y + h, x:x + w]

    if cropped_image.size == 0:
        raise ValueError("Selected region has no pixels. Please adjust the region coordinates.")

    # Reshape for clustering
    pixels = cropped_image.reshape(-1, 3)

    # Perform KMeans clustering to find dominant color
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]

    # Convert from BGR to HEX
    dominant_color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(dominant_color[2]), int(dominant_color[1]), int(dominant_color[0])
    )
    return dominant_color_hex

# Streamlit UI
st.title("Shirt and Pant Color Matcher")

# Image upload
uploaded_file = st.file_uploader("Upload an image (JPEG/PNG):", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Invalid image file. Please upload a valid image.")
        else:
            # Convert image to grayscale for detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize and preprocess image for better detection
            height, width = gray_image.shape[:2]
            if width > 500:
                scaling_factor = 500 / width
                gray_image = cv2.resize(gray_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            
            gray_image = cv2.equalizeHist(gray_image)

            # Detect upper body (shirt) and lower body (pants)
            upper_body = upper_body_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3)
            lower_body = lower_body_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3)

            # Debug visualization: Draw detected regions
            for (x, y, w, h) in upper_body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for shirt
            for (x, y, w, h) in lower_body:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for pants

            # Display the uploaded image with detected regions
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Regions", use_container_width=True)

            # Extract colors for shirt and pants
            shirt_color = "Not detected"
            pant_color = "Not detected"

            if len(upper_body) > 0:
                shirt_region = upper_body[0]  # Take the first detected region
                shirt_color = get_dominant_color(image, shirt_region)
            else:
                st.warning("Upper body not detected. You can select manually below.")

            if len(lower_body) > 0:
                pant_region = lower_body[0]  # Take the first detected region
                pant_color = get_dominant_color(image, pant_region)
            else:
                st.warning("Lower body not detected. You can select manually below.")

            # Allow manual region selection if detection fails
            if st.button("Manually Select Regions"):
                shirt_region = cv2.selectROI("Select Shirt Region", image, fromCenter=False, showCrosshair=True)
                shirt_color = get_dominant_color(image, shirt_region)

                pant_region = cv2.selectROI("Select Pant Region", image, fromCenter=False, showCrosshair=True)
                pant_color = get_dominant_color(image, pant_region)

            # Display detected colors
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Shirt Color:** {shirt_color}")
                if shirt_color != "Not detected":
                    st.markdown(
                        f"""<div style="background-color:{shirt_color}; width:100px; height:30px; border:1px solid #000;"></div>""",
                        unsafe_allow_html=True,
                    )
            with col2:
                st.markdown(f"**Pant Color:** {pant_color}")
                if pant_color != "Not detected":
                    st.markdown(
                        f"""<div style="background-color:{pant_color}; width:100px; height:30px; border:1px solid #000;"></div>""",
                        unsafe_allow_html=True,
                    )

            # Check if both shirt and pant colors were detected
            if shirt_color != "Not detected" and pant_color != "Not detected":
                # Check if the combination matches the dataset
                match_found = (
                    ((data[color1_col] == shirt_color) & (data[color2_col] == pant_color)) |
                    ((data[color1_col] == pant_color) & (data[color2_col] == shirt_color))
                ).any()

                # Display the match result
                if match_found:
                    st.success(f"The combination of shirt color '{shirt_color}' and pant color '{pant_color}' is a match!")
                else:
                    st.warning(f"The combination of shirt color '{shirt_color}' and pant color '{pant_color}' does not match.")
            else:
                st.error("Could not detect both shirt and pant colors. Please use manual selection or upload a clearer image.")

    except ValueError as ve:
        st.error(f"Error processing region: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload an image file.")
