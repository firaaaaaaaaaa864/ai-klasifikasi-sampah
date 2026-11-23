import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="AI Klasifikasi Sampah 12 Kelas", page_icon="♻️", layout="wide")
st.title("♻️ Sistem Identifikasi Jenis Sampah Berbasis CNN")
st.write("Unggah foto sampah (Maksimal 5 gambar sekaligus) untuk mendeteksi jenis, kategori kelas, dan cara penanganannya.")

# 2. Load Model AI dari Google Colab
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_sampah_12kelas.h5')

model = load_model()

# 3. Kamus 12 Kelas Sampah
INFO_SAMPAH = {
    "battery": {
        "nama": "Baterai Bekas",
        "kelas_utama": "B3 (Bahan Berbahaya & Beracun)",
        "keterangan": "Jangan dibuang ke tempat sampah biasa karena mengandung logam berat. Kumpulkan di kotak khusus limbah elektronik (e-waste)."
    },
    "biological": {
        "nama": "Sampah Organik / Sisa Makanan",
        "kelas_utama": "Organik (Kompos)",
        "keterangan": "Dapat diolah menjadi pupuk kompos, eco-enzyme, atau pakan maggot BSF."
    },
    "brown-glass": {
        "nama": "Kaca Cokelat",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Pisahkan berdasarkan warna kaca dan pastikan bersih sebelum disetorkan ke bank sampah."
    },
    "cardboard": {
        "nama": "Kardus / Karton",
        "kelas_utama": "Anorganik (Daur Ulang Kertas)",
        "keterangan": "Lipat hingga pipih dan pastikan tetap kering agar nilai jual daur ulangnya tinggi."
    },
    "clothes": {
        "nama": "Pakaian / Tekstil",
        "kelas_utama": "Anorganik (Guna Ulang / Daur Ulang Tekstil)",
        "keterangan": "Jika masih layak pakai dapat disumbangkan; jika rusak dapat didaur ulang menjadi kain perca atau isian bantal."
    },
    "green-glass": {
        "nama": "Kaca Hijau",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Botol/kaca berwarna hijau harus dipisahkan dari kaca bening saat proses peleburan daur ulang."
    },
    "metal": {
        "nama": "Logam / Kaleng",
        "kelas_utama": "Anorganik (Daur Ulang Logam)",
        "keterangan": "Bilas sisa isi kaleng dan pipihkan untuk menghemat ruang penyimpanan."
    },
    "paper": {
        "nama": "Kertas",
        "kelas_utama": "Anorganik (Daur Ulang Kertas)",
        "keterangan": "Dapat didaur ulang menjadi kertas baru selama tidak berminyak atau basah parah."
    },
    "plastic": {
        "nama": "Plastik",
        "kelas_utama": "Anorganik (Daur Ulang Plastik)",
        "keterangan": "Kosongkan cairan, remukkan botol, dan lepas label plastik jika memungkinkan."
    },
    "shoes": {
        "nama": "Sepatu / Alas Kaki",
        "kelas_utama": "Anorganik (Limbah Khusus / Guna Ulang)",
        "keterangan": "Donasikan jika masih layak, atau kirim ke program daur ulang sol karet/tekstil."
    },
    "trash": {
        "nama": "Residu / Sampah Umum",
        "kelas_utama": "Residu (TPA)",
        "keterangan": "Sampah campuran yang sulit didaur ulang dan harus dibuang ke tempat sampah residu menuju TPA."
    },
    "white-glass": {
        "nama": "Kaca Bening / Putih",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Kaca bening memiliki nilai daur ulang tertinggi; bungkus rapat jika dalam kondisi pecah."
    }
}

CLASS_NAMES = sorted(list(INFO_SAMPAH.keys()))

# 4. Fitur Upload Maksimal 5 Foto
uploaded_files = st.file_uploader(
    "Pilih gambar sampah (Format: JPG, JPEG, PNG — Maksimal 5 foto)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error(f"⚠️ Batas maksimal adalah 5 foto! Kamu mengunggah {len(uploaded_files)} foto. Silakan hapus beberapa foto terlebih dahulu.")
    else:
        st.success(f"Berhasil memuat {len(uploaded_files)} foto. Menganalisis...")
        
        for idx, file in enumerate(uploaded_files):
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            image = Image.open(file).convert("RGB")
            with col1:
                st.image(image, caption=f"Foto {idx+1}: {file.name}", use_container_width=True)
            
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            label_key = CLASS_NAMES[predicted_idx]
            detail = INFO_SAMPAH[label_key]
            
            with col2:
                st.subheader(f"🏷️ Jenis Sampah: {detail['nama']} ({label_key})")
                st.write(f"**Masuk Kelas:** `{detail['kelas_utama']}`")
                st.write(f"**Tingkat Keyakinan AI:** `{confidence:.2f}%`")
                st.info(f"**Keterangan Pengelolaan:** {detail['keterangan']}")