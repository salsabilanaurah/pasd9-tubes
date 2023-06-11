import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel(r'C:\Users\prase\Downloads\cosmetic_cleaned.xlsx')

# Set color palette
color_palette = ["#000000", "#808080", "#ffffff"]
st.set_page_config(page_title="Skin Sense", layout="wide", initial_sidebar_state="collapsed")

# Split dataset into training and testing sets
X = df[["Price", "Rank", "Combination", "Dry", "Normal", "Oily"]]
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the k-NN model to the training data
k = 5  # Jumlah tetangga terdekat yang ingin diambil
model = NearestNeighbors(n_neighbors=k)
model.fit(X_train)

def main():
    page = st.session_state.get("page")

    if page is None:
        show_home()
    elif page == "Recommendation":
        show_recommendation()
    elif page == "BrandRecommendation":
        show_brand_recommendation()
    elif page == "MLPrediction":
        show_ml_prediction()

def show_home():
    st.title("Selamat datang di Skin Sense")
    st.write("")

    # Button to start recommendation feature
    if st.button("Rekomendasi Produk untuk Kulitmu"):
        st.session_state["page"] = "Recommendation"
        st.experimental_rerun()

    # Button to start brand recommendation feature
    if st.button("Rekomendasi Produk dari Brand Favorite-mu"):
        st.session_state["page"] = "BrandRecommendation"
        st.experimental_rerun()

    # Button to start ML prediction feature
    if st.button("Rekomendasi Produk Menggunakan ML"):
        st.session_state["page"] = "MLPrediction"
        st.experimental_rerun()

def show_recommendation():
    st.title("Rekomendasi Produk untuk Kulitmu")

    # Back button
    if st.button("Back", key="back_button"):
        st.session_state.pop("page")
        st.experimental_rerun()

    st.write("Pilihlah satu jenis skincare yang ingin kamu lihat:")

    # Dropdown to select skincare type
    selected_label = st.selectbox("Tipe Skincare", df["Label"].unique())

    # Filter data based on selected label and sort by rank
    filtered_data = df[df["Label"] == selected_label].sort_values("Rank", ascending=False)

    # Display recommendation based on brand with highest rank
    recommended_brand = filtered_data.iloc[0]["Brand"]
    st.write("Rekomendasi:", recommended_brand)

    # Dropdown to select skin type
    selected_skin_type = st.selectbox("Jenis Kulit", ["Combination", "Dry", "Normal", "Oily"])

    # Filter data based on selected skin type and recommended brand
    filtered_data = filtered_data[filtered_data[selected_skin_type] == 1]

    # Display relevant data (Brand, Name, Price, Rank)
    st.write("Data Produk yang Sesuai:")
    st.dataframe(filtered_data[["Brand", "Name", "Price", "Rank"]])

def show_brand_recommendation():
    st.title("Rekomendasi Produk dari Brand Favorite-mu")

    # Back button
    if st.button("Back", key="back_button"):
        st.session_state.pop("page")
        st.experimental_rerun()

    st.write("Pilihlah Brand yang ingin kamu lihat:")

    # Dropdown to select brand
    selected_brand = st.selectbox("Brand", df["Brand"].unique())

    # Filter data based on selected brand and sort by rank
    filtered_data = df[df["Brand"] == selected_brand].sort_values("Rank", ascending=False)

    # Display relevant data (Brand, Label, Name, Price, Rank)
    st.write("Data Produk yang Sesuai:")
    st.dataframe(filtered_data[["Brand", "Label", "Name", "Price", "Rank"]])

def show_ml_prediction():
    st.title("Rekomendasi Produk Menggunakan ML")

    # Back button
    if st.button("Back", key="back_button"):
        st.session_state.pop("page")
        st.experimental_rerun()

    st.write("Masukkan atribut produk yang kamu miliki:")

    # Dropdown to select label
    selected_label = st.selectbox("Label", df["Label"].unique())

    # Dropdown for skin type
    selected_skin_type = st.selectbox("Jenis Kulit", ["Combination", "Dry", "Normal", "Oily"])

    # Set the skin type values based on user selection
    combination = 1 if selected_skin_type == "Combination" else 0
    dry = 1 if selected_skin_type == "Dry" else 0
    normal = 1 if selected_skin_type == "Normal" else 0
    oily = 1 if selected_skin_type == "Oily" else 0

    price = st.slider("Price", 3, 370)

    # Create a new data point based on user input
    new_data = [[price, combination, dry, normal, oily, 0]]

    # Get k nearest neighbors from the trained model using the testing data
    distances, indices = model.kneighbors(new_data)

    st.write("Hasil Rekomendasi Produk:")
    count = 1  # Inisialisasi nomor urut
    product_found = False  # Flag untuk menandakan apakah ada produk yang ditemukan
    for i in range(k):
        index = indices[0][i]
        if index not in X_test.index:
            continue

        product = df.iloc[X_test.index.get_loc(index), :]
        if product['Label'] == selected_label:
            product_found = True
            st.write(f"{count}. {product['Brand']} - {product['Name']}")
            count += 1

    if not product_found:
        st.write("Tidak menemukan produk tersebut untuk harga ini")

if __name__ == "__main__":
    main()
