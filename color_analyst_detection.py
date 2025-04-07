import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from mtcnn import MTCNN
from sklearn.cluster import KMeans
from PIL import Image
from skimage.color import deltaE_ciede2000

def detect_face(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        return image[y:y+h, x:x+w]
    return image

def dominant_colors(image, max_k=6):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    pixels = lab.reshape(-1, 3)
    inertia = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pixels)
        inertia.append(kmeans.inertia_)
    best_k = np.argmin(np.gradient(inertia)) + 2  
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(pixels)
    return kmeans.cluster_centers_.astype(int), best_k

def classify_season(colors):
    # Referensi warna undertone untuk tiap season
    ref_colors = {
        "Spring": [
            np.array([255, 200, 170]),  # Warm peachy
            np.array([250, 180, 140]),  # Light coral
            np.array([240, 160, 120])   # Golden beige
        ],
        "Summer": [
            np.array([220, 180, 170]),  # Cool soft pink
            np.array([200, 160, 150]),  # Rosy beige
            np.array([190, 150, 140])   # Muted mauve
        ],
        "Autumn": [
            np.array([180, 120, 90]),   # Warm tan
            np.array([160, 100, 70]),   # Bronze brown
            np.array([140, 80, 50])     # Deep golden
        ],
        "Winter": [
            np.array([220, 190, 170]),  # Cool beige
            np.array([190, 160, 140]),  # Neutral taupe
            np.array([170, 140, 120])   # Deep ash brown
        ]
    }

    # Menghitung warna rata-rata dari wajah
    avg_color = np.mean(colors, axis=0)
    
    # Menentukan season dengan perhitungan jarak warna paling kecil
    min_diff = float("inf")
    best_season = ""

    for season, ref_list in ref_colors.items():
        for ref in ref_list:
            diff = deltaE_ciede2000(avg_color, ref)
            if diff < min_diff:
                min_diff, best_season = diff, season
    
    return best_season

def get_color_recommendations(season):
    palettes = {
        "Spring": [(255, 223, 186), (255, 153, 102), (255, 204, 102)],
        "Summer": [(204, 229, 255), (153, 204, 255), (255, 204, 229)],
        "Autumn": [(204, 102, 0), (153, 76, 0), (255, 153, 51)],
        "Winter": [(0, 51, 102), (0, 102, 204), (102, 0, 153)]
    }
    return palettes.get(season, [])

def plot_palette(colors, title):
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.imshow([colors], aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return fig

def main():
    st.set_page_config(page_title="Advanced Color Analyst", page_icon="🎨", layout="wide")
    st.title("🌈 Advanced Color Analyst")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        face_image = detect_face(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width=250)
        with col2:
            st.image(face_image, caption="Detected Face", width=250)
        
        if st.button("Hasil Analisis"):
            colors, _ = dominant_colors(face_image)
            season = classify_season(colors)
            recommended_colors = get_color_recommendations(season)
            
            st.subheader(f"Detected Season: {season}")
            
            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(plot_palette(colors/255, "Dominant Colors"))
            with col4:
                st.pyplot(plot_palette(np.array(recommended_colors)/255, "Recommended Colors"))

if __name__ == "__main__":
    main()
