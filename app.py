import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_paste_button import paste_image_button
import io

# 1. Konfigurasi Halaman & Tema Warna Hijau-Biru (Custom CSS)
st.set_page_config(page_title="EcoSort AI - Klasifikasi Sampah", page_icon="♻️", layout="wide")

st.markdown("""
    <style>
    /* Latar belakang gradasi hijau mint ke biru muda lembut */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 50%, #e3f2fd 100%);
        color: #1b3b2b;
    }
    /* Kotak kartu hasil prediksi */
    .result-card {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #00897b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    /* Judul utama */
    h1 {
        color: #00695c !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("♻️ EcoSort AI: Sistem Identifikasi & Pengelolaan Sampah")
st.write("Unggah file foto **atau** tempel (Copy-Paste) gambar sampah dari internet (**Maksimal 5 foto**).")

# 2. Load Model AI
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_sampah_12kelas.h5')

model = load_model()

# 3. Kamus 12 Kelas Sampah + Link YouTube & Platform Indonesia
INFO_SAMPAH = {
    "battery": {
        "nama": "Baterai Bekas",
        "kelas_utama": "B3 (Bahan Berbahaya & Beracun)",
        "keterangan": "Mengandung logam berat berbahaya. Jangan dibuang ke tempat sampah biasa atau dibakar.",
        "aksi_judul": "📍 Tempat Penyaluran Limbah B3 / Elektronik di Indonesia:",
        "link_aksi_nama": "Drop Box E-Waste RJ (Jakarta & Sekitarnya) / Dinas LH Kota",
        "link_aksi_url": "https://ewasterj.com/",
        "link_video_nama": "▶️ Video Cara Aman Menyimpan & Menyalurkan Baterai Bekas",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+membuang+limbah+baterai+bekas+yang+benar"
    },
    "biological": {
        "nama": "Sampah Organik / Sisa Makanan",
        "kelas_utama": "Organik (Dapat Terurai)",
        "keterangan": "Dapat diolah menjadi kompos padat, pupuk cair (Eco-Enzyme), atau pakan maggot BSF.",
        "aksi_judul": "🌱 Komunitas & Pengolahan Organik Indonesia:",
        "link_aksi_nama": "Komunitas Eco-Enzyme Nusantara / Waste4Change",
        "link_aksi_url": "https://waste4change.com/",
        "link_video_nama": "▶️ Video Tutorial Membuat Kompos & Eco-Enzyme di Rumah",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+membuat+kompos+dan+eco+enzyme+dari+sampah+dapur"
    },
    "brown-glass": {
        "nama": "Kaca Cokelat",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Pisahkan dari kaca bening, bersihkan sisa cairan, lalu setorkan ke bank sampah.",
        "aksi_judul": "💰 Jual / Setor Sampah Kaca di Indonesia:",
        "link_aksi_nama": "Aplikasi MallSampah (Jemput & Jual Sampah Daur Ulang)",
        "link_aksi_url": "https://www.mallsampah.com/",
        "link_video_nama": "▶️ Video Ide Kerajinan & Daur Ulang Botol Kaca",
        "link_video_url": "https://www.youtube.com/results?search_query=ide+daur+ulang+botol+kaca+bekas"
    },
    "cardboard": {
        "nama": "Kardus / Karton",
        "kelas_utama": "Anorganik (Daur Ulang Kertas)",
        "keterangan": "Lipat hingga pipih dan pastikan kondisi kering agar bernilai jual tinggi saat disetor.",
        "aksi_judul": "💰 Jual Kardus Bekas ke Pengepul / Aplikasi Daur Ulang:",
        "link_aksi_nama": "Duitin Indonesia / MallSampah",
        "link_aksi_url": "https://duitin.id/",
        "link_video_nama": "▶️ Video Tutorial Mengolah Kardus Bekas Jadi Barang Berguna",
        "link_video_url": "https://www.youtube.com/results?search_query=ide+kreatif+daur+ulang+kardus+bekas"
    },
    "clothes": {
        "nama": "Pakaian / Tekstil",
        "kelas_utama": "Anorganik (Guna Ulang / Daur Ulang Tekstil)",
        "keterangan": "Pakaian layak pakai bisa didonasikan; pakaian rusak bisa didaur ulang menjadi kain perca/benang baru.",
        "aksi_judul": "🤝 Platform Donasi & Daur Ulang Pakaian di Indonesia:",
        "link_aksi_nama": "Kitabisa.com (Donasi Pakaian) / Setali Indonesia (Daur Ulang Baju)",
        "link_aksi_url": "https://kitabisa.com/explore/panti-asuhan",
        "link_video_nama": "▶️ Video Ide Mengubah Baju Bekas Menjadi Tas / Keset",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+mendaur+ulang+baju+bekas+menjadi+barang+berguna"
    },
    "green-glass": {
        "nama": "Kaca Hijau",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Pisahkan kaca hijau dari warna kaca lainnya sebelum diserahkan ke bank sampah.",
        "aksi_judul": "💰 Jual / Setor Sampah Kaca di Indonesia:",
        "link_aksi_nama": "Aplikasi MallSampah / Bank Sampah Terdekat",
        "link_aksi_url": "https://www.mallsampah.com/",
        "link_video_nama": "▶️ Video Ide Kreatif Botol Kaca Bekas",
        "link_video_url": "https://www.youtube.com/results?search_query=diy+botol+kaca+bekas+estetik"
    },
    "metal": {
        "nama": "Logam / Kaleng",
        "kelas_utama": "Anorganik (Daur Ulang Logam)",
        "keterangan": "Bilas sisa makanan/minuman dan pipihkan kaleng untuk menghemat tempat.",
        "aksi_judul": "💰 Jual Kaleng & Logam Bekas di Indonesia:",
        "link_aksi_nama": "Duitin Indonesia (Jual Sampah Terpilah)",
        "link_aksi_url": "https://duitin.id/",
        "link_video_nama": "▶️ Video Tutorial Daur Ulang Kaleng Bekas",
        "link_video_url": "https://www.youtube.com/results?search_query=ide+kreatif+dari+kaleng+bekas"
    },
    "paper": {
        "nama": "Kertas",
        "kelas_utama": "Anorganik (Daur Ulang Kertas)",
        "keterangan": "Kumpulkan kertas HVS, koran, atau buku bekas dalam kondisi kering dan tidak berminyak.",
        "aksi_judul": "💰 Jual Kertas Bekas di Indonesia:",
        "link_aksi_nama": "MallSampah / Duitin Indonesia",
        "link_aksi_url": "https://www.mallsampah.com/",
        "link_video_nama": "▶️ Video Cara Membuat Kertas Daur Ulang Sendiri",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+mendaur+ulang+kertas+bekas+menjadi+kertas+baru"
    },
    "plastic": {
        "nama": "Plastik",
        "kelas_utama": "Anorganik (Daur Ulang Plastik)",
        "keterangan": "Kosongkan air, remukkan botol/gelas plastik, dan pisahkan tutupnya jika berbeda jenis plastik.",
        "aksi_judul": "💰 Jual Botol & Gelas Plastik Bekas di Indonesia:",
        "link_aksi_nama": "Duitin Indonesia / Plasticpay",
        "link_aksi_url": "https://duitin.id/",
        "link_video_nama": "▶️ Video Tutorial Daur Ulang Botol & Gelas Plastik Bekas",
        "link_video_url": "https://www.youtube.com/results?search_query=ide+daur+ulang+botol+dan+gelas+plastik+bekas"
    },
    "shoes": {
        "nama": "Sepatu / Alas Kaki",
        "kelas_utama": "Anorganik (Guna Ulang / Limbah Khusus)",
        "keterangan": "Sepatu layak pakai dapat didonasikan ke panti asuhan atau komunitas sosial.",
        "aksi_judul": "🤝 Salurkan Donasi Sepatu Layak Pakai di Indonesia:",
        "link_aksi_nama": "Kitabisa - Donasi Panti Asuhan & Pelosok Negeri",
        "link_aksi_url": "https://kitabisa.com/explore/panti-asuhan",
        "link_video_nama": "▶️ Video Cara Membersihkan & Merestorasi Sepatu Lama",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+mencuci+dan+memperbaiki+sepatu+bekas"
    },
    "trash": {
        "nama": "Residu / Sampah Umum",
        "kelas_utama": "Residu (TPA)",
        "keterangan": "Sampah yang sulit didaur ulang (seperti tisu kotor, popok, styrofoam kotor). Buang ke tempat sampah residu.",
        "aksi_judul": "ℹ️ Edukasi Pengelolaan Sampah Residu:",
        "link_aksi_nama": "Waste4Change - Kelola Sampah Bertanggung Jawab",
        "link_aksi_url": "https://waste4change.com/",
        "link_video_nama": "▶️ Video Cara Mengurangi Sampah Residu Sehari-hari",
        "link_video_url": "https://www.youtube.com/results?search_query=cara+mengurangi+sampah+residu+rumah+tangga"
    },
    "white-glass": {
        "nama": "Kaca Bening / Putih",
        "kelas_utama": "Anorganik (Daur Ulang Kaca)",
        "keterangan": "Kaca bening memiliki nilai daur ulang tinggi. Bungkus rapat dengan kardus jika dalam kondisi pecah.",
        "aksi_judul": "💰 Jual / Setor Botol Kaca Bening di Indonesia:",
        "link_aksi_nama": "MallSampah Indonesia",
        "link_aksi_url": "https://www.mallsampah.com/",
        "link_video_nama": "▶️ Video Ide Daur Ulang Botol Kaca Bening",
        "link_video_url": "https://www.youtube.com/results?search_query=ide+kreatif+daur+ulang+botol+kaca+bening"
    }
}

CLASS_NAMES = sorted(list(INFO_SAMPAH.keys()))

# 4. Penyimpanan Sementara untuk Gambar Hasil Copy-Paste
if "pasted_images" not in st.session_state:
    st.session_state.pasted_images = []

# 5. Area Input: Tab Upload File & Tab Copy-Paste Gambar
tab1, tab2 = st.tabs(["📁 Unggah File Gambar", "📋 Copy-Paste Gambar (Ctrl+V)"])

uploaded_files = []
with tab1:
    files_from_disk = st.file_uploader(
        "Pilih gambar dari perangkat (JPG, JPEG, PNG — Maksimal 5 foto)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if files_from_disk:
        uploaded_files = files_from_disk

with tab2:
    st.write("Salin gambar dari Google (Klik kanan gambar > *Copy image*), lalu klik tombol di bawah ini:")
    col_paste, col_clear = st.columns([2, 1])
    with col_paste:
        paste_result = paste_image_button(
            label="📋 Tempel Gambar dari Clipboard",
            background_color="#00897b",
            hover_background_color="#00695c",
            errors="ignore"
        )
    with col_clear:
        if st.button("🗑️ Hapus Semua Tempelan"):
            st.session_state.pasted_images = []
            st.rerun()

    if paste_result.image_data is not None:
        # Cegah duplikasi gambar yang sama persis saat halaman refresh
        img_bytes = io.BytesIO()
        paste_result.image_data.save(img_bytes, format="PNG")
        img_val = img_bytes.getvalue()
        if not any(item["bytes"] == img_val for item in st.session_state.pasted_images):
            if len(st.session_state.pasted_images) < 5:
                st.session_state.pasted_images.append({
                    "image": paste_result.image_data.convert("RGB"),
                    "bytes": img_val,
                    "name": f"Gambar_Tempel_{len(st.session_state.pasted_images)+1}.png"
                })
            else:
                st.warning("⚠️ Sudah mencapai batas maksimal 5 gambar tempelan!")

# Gabungkan daftar gambar yang akan diproses
daftar_gambar = []
for f in uploaded_files:
    daftar_gambar.append({"image": Image.open(f).convert("RGB"), "name": f.name})
for p in st.session_state.pasted_images:
    daftar_gambar.append({"image": p["image"], "name": p["name"]})

# 6. Proses Prediksi & Tampilkan Hasil
if daftar_gambar:
    if len(daftar_gambar) > 5:
        st.error(f"⚠️ Batas maksimal adalah 5 foto sekaligus! Saat ini ada {len(daftar_gambar)} foto. Kurangi pilihanmu terlebih dahulu.")
    else:
        st.success(f"Menganalisis {len(daftar_gambar)} gambar sampah...")
        
        for idx, item in enumerate(daftar_gambar):
            image = item["image"]
            
            # Preprocessing
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            label_key = CLASS_NAMES[predicted_idx]
            detail = INFO_SAMPAH[label_key]
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2.3])
            
            with col1:
                st.image(image, caption=f"Foto {idx+1}: {item['name']}", use_container_width=True)
                
            with col2:
                st.subheader(f"🏷️ {detail['nama']} ({label_key})")
                st.markdown(f"**Kategori Kelas:** `{detail['kelas_utama']}` | **Tingkat Keyakinan AI:** `{confidence:.2f}%`")
                
                if confidence < 65.0:
                    st.warning("⚠️ *Catatan: Tingkat keyakinan AI rendah. Objek mungkin merupakan kerajinan daur ulang yang bentuknya sudah berubah dari sampah mentah.*")
                
                st.write(f"**Saran Pengelolaan:** {detail['keterangan']}")
                st.markdown(f"**{detail['aksi_judul']}** [{detail['link_aksi_nama']}]({detail['link_aksi_url']})")
                st.markdown(f"**Tutorial Pengolahan:** [{detail['link_video_nama']}]({detail['link_video_url']})")
            st.markdown('</div>', unsafe_allow_html=True)
