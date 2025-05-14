import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Fungsi untuk load model
@st.cache_resource
def load_garbage_model():
    model = load_model('model_classification_sampah.h5')
    return model

model = load_garbage_model()

# Label dan deskripsi
label_mapping = {
    "battery": "Baterai",
    "biological": "Sampah Biologis",
    "brown-glass": "Kaca Coklat",
    "cardboard": "Kardus",
    "clothes": "Kain/Pakaian",
    "green-glass": "Kaca Hijau",
    "metal": "Logam",
    "paper": "Kertas",
    "plastic": "Plastik",
    "shoes": "Sepatu",
    "trash": "Sampah Campuran",
    "white-glass": "Kaca Putih"
}

kategori_organik = ["Sampah Biologis", "Kertas", "Kardus"]
kategori_non_organik = ["Baterai", "Kaca Coklat", "Kaca Hijau", "Kaca Putih", "Plastik", "Logam", "Kain/Pakaian", "Sepatu", "Sampah Campuran"]

deskripsi_sampah = {
    "Baterai": "Mengandung bahan kimia berbahaya dan harus dibuang di tempat khusus.",
    "Sampah Biologis": "Sampah organik seperti sisa makanan, daun, dan bahan alami lainnya.",
    "Kaca Coklat": "Jenis kaca yang biasanya digunakan untuk botol bir atau minuman lainnya.",
    "Kardus": "Bahan daur ulang yang umum digunakan untuk kemasan dan harus dibuang ke tempat daur ulang.",
    "Kain/Pakaian": "Bisa didaur ulang atau disumbangkan jika masih layak pakai.",
    "Kaca Hijau": "Biasanya digunakan untuk botol kaca hijau seperti botol minuman.",
    "Logam": "Bisa didaur ulang menjadi berbagai produk baru seperti kaleng atau bagian kendaraan.",
    "Kertas": "Dapat didaur ulang menjadi kertas baru atau produk berbasis kertas lainnya.",
    "Plastik": "Bisa didaur ulang, tetapi butuh waktu lama untuk terurai di lingkungan.",
    "Sepatu": "Biasanya terbuat dari campuran bahan yang sulit didaur ulang, bisa disumbangkan jika masih layak pakai.",
    "Sampah Campuran": "Sampah yang tidak dapat dikategorikan ke dalam kelompok spesifik.",
    "Kaca Putih": "Kaca bening yang sering digunakan dalam kemasan makanan dan minuman."
}

class_names = list(label_mapping.keys())

# Navigasi Sidebar
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Prediksi Sampah", "Tentang"])

# Halaman Beranda
if page == "Beranda":
    st.markdown("<h1 style='text-align: center; color: green;'>♻️ Aplikasi Deteksi Sampah</h1>", unsafe_allow_html=True)
    st.markdown("### Selamat datang!")
    st.write("Aplikasi ini membantu kamu mengenali jenis sampah dari gambar dan memberi informasi penting seperti:")
    st.markdown("- ✅ Jenis sampah (organik / non-organik)")
    st.markdown("- 📄 Penjelasan kategori sampah")
    st.markdown("- 🧠 Edukasi singkat tentang daur ulang")
    st.image("https://images.unsplash.com/photo-1582719478175-ff3f767b89a7", caption="Ilustrasi Sampah", use_column_width=True)

    st.markdown("### 📷 Contoh Gambar")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Cardboard_box.jpg", caption="Kardus")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/Plastic_bottles.jpg", caption="Plastik")
    with col3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9e/Battery_collection.jpg", caption="Baterai")

    with st.expander("❓ Apa itu sampah organik dan non-organik?"):
        st.write("""
        **Sampah organik** adalah sampah yang berasal dari makhluk hidup dan bisa terurai secara alami, seperti daun, sisa makanan, atau kertas.
        
        **Sampah non-organik** berasal dari benda tak hidup dan sulit terurai, seperti plastik, kaca, logam, dan baterai.
        """)

# Halaman Prediksi
elif page == "Prediksi Sampah":
    st.markdown("## 🧪 Deteksi Jenis Sampah dari Gambar")
    st.write("Upload gambar sampah untuk mengetahui jenis dan penjelasannya.")

    uploaded_file = st.file_uploader("Unggah gambar sampah...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        try:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_prob = np.max(predictions)

            THRESHOLD = 0.7

            if predicted_prob < THRESHOLD:
                st.warning("🔍 Gambar tidak dikenali dalam dataset.")
            else:
                kelas_inggris = class_names[predicted_index]
                kelas_indonesia = label_mapping.get(kelas_inggris, "Tidak Diketahui")
                kategori = "Organik" if kelas_indonesia in kategori_organik else "Non-Organik"
                deskripsi = deskripsi_sampah.get(kelas_indonesia, "Tidak ada deskripsi.")

                st.success(f"✅ Prediksi: **{kelas_indonesia}** (Prob: {predicted_prob:.2f})")
                st.info(f"🗑️ Kategori: {kategori}")
                st.markdown(f"📄 **Deskripsi:** {deskripsi}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

# Halaman Tentang
elif page == "Tentang":
    st.markdown("## ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan model deep learning berbasis Convolutional Neural Network (CNN) untuk mengenali jenis sampah dari gambar.
    
    Model telah dilatih menggunakan dataset dari berbagai kategori sampah, baik organik maupun non-organik.
    
    **Tujuan**:
    - Meningkatkan kesadaran memilah sampah
    - Membantu masyarakat dalam edukasi daur ulang
    - Mengurangi pencemaran dengan pemilahan yang benar
    """)
    st.markdown("📚 Dataset: [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)")
    st.markdown("🧑‍💻 Dibuat oleh: **[Namamu]**")

# Footer
st.markdown("---")
st.markdown("<center>© 2025 - Aplikasi Deteksi Sampah oleh [Namamu]</center>", unsafe_allow_html=True)
