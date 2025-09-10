import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

# ---------------- Setup ----------------
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load embeddings
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load styles dataset
styles_df = pd.read_csv("styles.csv", on_bad_lines="skip")

# Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tensorflow.keras.Sequential([base_model, GlobalMaxPooling2D()])

st.title('Outfit Recommendation System')

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #1a1a2e, #3b1c32, #0f3460, #16213e);
        background-size: 400% 400%;
        animation: gradientMove 20s ease infinite;
    }
    .card {
        background: rgba(30, 30, 30, 0.6);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: floatUp 0.6s ease forwards;
        opacity: 0;
    }
    @keyframes floatUp {
        from { transform: translateY(40px); opacity: 0; }
        to { transform: translateY(0px); opacity: 1; }
    }
    img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Session state ----------------
if "wishlist" not in st.session_state:
    st.session_state["wishlist"] = []
if "query_features" not in st.session_state:
    st.session_state["query_features"] = None
if "query_image" not in st.session_state:
    st.session_state["query_image"] = None
if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = None

# ---------------- Functions ----------------
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"File upload error: {e}")
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, n_results=6):
    similarities = np.dot(feature_list, features)
    sorted_idx = np.argsort(similarities)[::-1]
    top_indices = sorted_idx[:n_results]
    top_sims = similarities[top_indices] * 100
    return top_sims, top_indices

# ---------------- Sidebar Filters ----------------
with st.sidebar.form("filters_form"):
    st.subheader("🔍 Filters")

    gender_filter = st.selectbox("Select gender", ["All"] + styles_df["gender"].dropna().unique().tolist())
    master_filter = st.selectbox("Select masterCategory", ["All"] + styles_df["masterCategory"].dropna().unique().tolist())
    subcat_filter = st.selectbox("Select subCategory", ["All"] + styles_df["subCategory"].dropna().unique().tolist())
    article_filter = st.selectbox("Select articleType", ["All"] + styles_df["articleType"].dropna().unique().tolist())
    colour_filter = st.selectbox("Select baseColour", ["All"] + styles_df["baseColour"].dropna().unique().tolist())
    season_filter = st.selectbox("Select season", ["All"] + styles_df["season"].dropna().unique().tolist())
    year_filter = st.selectbox("Select year", ["All"] + styles_df["year"].dropna().unique().astype(str).tolist())
    usage_filter = st.selectbox("Select usage", ["All"] + styles_df["usage"].dropna().unique().tolist())
    product_filter = st.selectbox("Select productDisplayName", ["All"] + styles_df["productDisplayName"].dropna().unique().tolist())

    search_btn = st.form_submit_button("Search")

# ---------------- Uploader ----------------
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        if (st.session_state["uploaded_file_path"] is None) or (file_path != st.session_state["uploaded_file_path"]):
            st.session_state["uploaded_file_path"] = file_path
            st.session_state["query_features"] = feature_extraction(file_path, model)
            st.session_state["query_image"] = file_path

# ---------------- Show Query Image ----------------
if st.session_state["query_features"] is not None:
    st.subheader("⭐ Current Query Image")
    try:
        st.image(st.session_state["query_image"], caption="Current Query", use_container_width=False)
    except Exception:
        pass

    # Instant recommendations after upload
    sims, indices = recommend(st.session_state["query_features"], feature_list)
    st.subheader("🎨 Recommended Outfits")
    cols = st.columns(3)
    for pos, idx in enumerate(indices[:9]):
        with cols[pos % 3]:
            st.image(filenames[idx], caption=f"Similarity: {sims[pos]:.1f}%")
            if st.button("Add to Wishlist", key=f"wish_{idx}"):
                if filenames[idx] not in st.session_state["wishlist"]:
                    st.session_state["wishlist"].append(filenames[idx])
            if st.button("More Like This", key=f"more_{idx}"):
                st.session_state["query_features"] = feature_extraction(filenames[idx], model)
                st.session_state["query_image"] = filenames[idx]
                st.rerun()

# ---------------- Handle Search / Filters ----------------
if search_btn:
    filtered_df = styles_df.copy()

    # Apply filters
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["gender"] == gender_filter]
    if master_filter != "All":
        filtered_df = filtered_df[filtered_df["masterCategory"] == master_filter]
    if subcat_filter != "All":
        filtered_df = filtered_df[filtered_df["subCategory"] == subcat_filter]
    if article_filter != "All":
        filtered_df = filtered_df[filtered_df["articleType"] == article_filter]
    if colour_filter != "All":
        filtered_df = filtered_df[filtered_df["baseColour"] == colour_filter]
    if season_filter != "All":
        filtered_df = filtered_df[filtered_df["season"] == season_filter]
    if year_filter != "All":
        filtered_df = filtered_df[filtered_df["year"] == int(year_filter)]
    if usage_filter != "All":
        filtered_df = filtered_df[filtered_df["usage"] == usage_filter]
    if product_filter != "All":
        filtered_df = filtered_df[filtered_df["productDisplayName"] == product_filter]

    # Case 1: Image uploaded → recommend + filter
    if st.session_state["query_features"] is not None:
        sims, indices = recommend(st.session_state["query_features"], feature_list)

        matched_files = []
        for idx in indices:
            file_id = int(os.path.basename(filenames[idx]).split('.')[0])  # assumes filename is id.jpg
            if file_id in filtered_df["id"].values:
                matched_files.append((filenames[idx], sims[indices.tolist().index(idx)]))

        if matched_files:
            st.subheader("🎨 Filtered Recommendations")
            cols = st.columns(3)
            for pos, (f, sim) in enumerate(matched_files[:9]):
                with cols[pos % 3]:
                    st.image(f, caption=f"Similarity: {sim:.1f}%")
                    if st.button("Add to Wishlist", key=f"wish_filter_{pos}"):
                        if f not in st.session_state["wishlist"]:
                            st.session_state["wishlist"].append(f)
                    if st.button("More Like This", key=f"more_filter_{pos}"):
                        st.session_state["query_features"] = feature_extraction(f, model)
                        st.session_state["query_image"] = f
                        st.rerun()
        else:
            st.warning("⚠️ No results match your filters. Try changing them.")

    # Case 2: No image uploaded → just filters
    else:
        if not filtered_df.empty:
            st.subheader("🎨 Items Matching Filters")
            cols = st.columns(3)
            for pos, row in enumerate(filtered_df.head(9).itertuples()):
                img_path = os.path.join("images", str(row.id) + ".jpg")
                if os.path.exists(img_path):
                    with cols[pos % 3]:
                        st.image(img_path, caption=row.productDisplayName)
        else:
            st.warning("⚠️ No results match your filters. Try changing them.")

# ---------------- Wishlist ----------------
if st.session_state["wishlist"]:
    st.subheader("⭐ Your Wishlist")
    wish_cols = st.columns(3)
    for i, w_item in enumerate(st.session_state["wishlist"]):
        with wish_cols[i % 3]:
            st.image(w_item)
    if st.button("Clear Wishlist"):
        st.session_state["wishlist"] = []
        st.success("Wishlist cleared")
