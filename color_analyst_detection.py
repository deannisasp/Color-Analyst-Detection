import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    else:
        return image

def extract_skin_tone(image, k=5):
    face_image = detect_face(image)
    hsv = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 11, 50], dtype=np.uint8)
    upper_skin = np.array([35, 95, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = face_image[skin_mask > 0]
    if len(skin_pixels) > 0:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(skin_pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return face_image, colors, kmeans.labels_
    else:
        return face_image, np.array([[255, 224, 189]]), None

def classify_season(colors, labels):
    h_values = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0][0] for color in colors]
    counts = np.bincount(labels, minlength=len(colors))
    weights = counts / np.sum(counts)  # Normalisasi bobot berdasarkan jumlah piksel
    avg_hue = np.dot(h_values, weights)
    
    if avg_hue < 15 or avg_hue > 160:
        return "Autumn" if np.dot(colors.mean(axis=1), weights) < 150 else "Spring"
    elif 15 <= avg_hue < 40:
        return "Spring"
    elif 40 <= avg_hue < 75:
        return "Summer"
    else:
        return "Winter"

def get_color_recommendations(season):
    color_palettes = {
        "Spring": [(255, 223, 186), (255, 153, 102), (255, 204, 102), (102, 255, 178), (102, 204, 255)],
        "Summer": [(204, 229, 255), (153, 204, 255), (255, 204, 229), (255, 153, 204), (153, 255, 204)],
        "Autumn": [(204, 102, 0), (153, 76, 0), (255, 153, 51), (153, 102, 51), (102, 51, 0)],
        "Winter": [(0, 51, 102), (0, 102, 204), (102, 0, 153), (204, 0, 102), (102, 204, 255)]
    }
    return color_palettes.get(season, [])

def main():
    st.set_page_config(page_title="Color Analyst", page_icon="🎨", layout="wide")
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #ff6600;
        }
        .result-box {
            padding: 10px;
            border-radius: 10px;
            background-color: #f5f5f5;
            text-align: center;
            color: #333333;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>Color Analyst</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], help="Upload a clear image of your face.")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        if st.button("Hasil Analisis"):
            face_image, colors, labels = extract_skin_tone(image, k=5)
            if labels is not None:
                season = classify_season(colors, labels)
                recommended_colors = get_color_recommendations(season)
            else:
                season = "Unknown"
                recommended_colors = []
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded Image", width=250)
            with col2:
                st.image(face_image, caption="Extracted Face", width=250)
            
            st.markdown(f"<div class='result-box'><h2>Detected Season: {season}</h2></div>", unsafe_allow_html=True)
            
            col3, col4 = st.columns([1, 1])
            with col3:
                st.subheader("Extracted Skin Colors")
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.bar(range(len(colors)), [1]*len(colors), color=[col/255 for col in colors], edgecolor='black')
                ax.set_xticks(range(len(colors)))
                ax.set_xticklabels([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], rotation=45)
                ax.set_xlabel("Dominant Skin Tones")
                st.pyplot(fig)
            
            with col4:
                st.subheader("Recommended Colors")
                fig, ax = plt.subplots(figsize=(4, 2))
                ax.bar(range(len(recommended_colors)), [1]*len(recommended_colors), color=[np.array(col)/255 for col in recommended_colors], edgecolor='black')
                ax.set_xticks(range(len(recommended_colors)))
                ax.set_xticklabels([f'#{r:02x}{g:02x}{b:02x}' for r, g, b in recommended_colors], rotation=45)
                ax.set_xlabel("Recommended Colors")
                st.pyplot(fig)
            
if __name__ == "__main__":
    main()
