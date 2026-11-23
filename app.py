import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_paste_button import paste_image_button
import io

# 1. Konfigurasi Halaman & Tema Warna Hijau-Biru Terang
st.set_page_config(page_title="EcoSort AI - Klasifikasi Sampah", page_icon="♻️", layout="wide")

st.markdown("""
    <style>
    /* Latar belakang gradasi hijau mint ke biru lembut */
    .stApp {
        background: linear-gradient(135deg, #d4f1f4 0%, #d8f3dc 50%, #cdeac0 100%) !important;
    }
    /* Paksa semua teks biasa berwarna gelap agar terbaca jelas */
    .stApp p, .stApp label, .stApp li, .stApp span {
        color: #102a43 !important;
    }
    h1, h2, h3 {
        color: #005f73 !important;
    }
    
    /* === CERAHKAN AREA UPLOAD & HILANGKAN BLOK HITAM === */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #0a9396 !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }
    section[data-testid="stFileUploaderDropzone"] div,
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] small {
        color: #005f73 !important;
    }
    /* Tombol 'Upload' / 'Browse files' dibuat cerah hijau toska dengan tulisan PUTIH TERANG */
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #00897b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: bold !important;
    }
    section[data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #00695c !important;
    }
    section[data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
    }

    /* === CERAHKAN TOMBOL + TAMBAH GAMBAR (POPOVER) DENGAN TULISAN BIRU TERANG === */
    button[data-testid="stPopoverButton"],
    div[data-testid="stPopover"] button {
        background-color: #ffffff !important;
        border: 2.5px solid #0077b6 !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(0, 119, 182, 0.18) !important;
    }
    button[data-testid="stPopoverButton"] p,
    button[data-testid="stPopoverButton"] span,
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button span {
        color: #0077b6 !important;
        font-size: 1.05em !important;
        font-weight: 800 !important;
    }
    button[data-testid="stPopoverButton"]:hover,
    div[data-testid="stPopover"] button:hover {
        background-color: #e0f7fa !important;
        border-color: #0096c7 !important;
    }
    /* Isi kotak popup saat tombol + diklik */
    div[data-testid="stPopoverBody"] {
        background-color: #f0fdf4 !important;
        border: 2px solid #0a9396 !important;
        border-radius: 14px !important;
    }
    /* Hilangkan blok hitam di sekitar tombol Paste */
    iframe[title="streamlit_paste_button.paste_image_button"] {
        background-color: transparent !important;
    }

    /* === CERAHKAN KAMERA === */
    div[data-testid="stCameraInput"] {
        background-color: #ffffff !important;
        padding: 12px !important;
        border-radius: 14px !important;
        border: 2px solid #0a9396 !important;
    }
    div[data-testid="stCameraInput"] button {
        background-color: #00897b !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }
    div[data-testid="stCameraInput"] button * {
        color: #ffffff !important;
    }

    /* Tag kategori hijau mint lembut */
    code {
        background-color: #e0f2f1 !important;
        color: #004d40 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    /* Kotak peringatan keraguan kustom berwarna kuning cerah */
    .warning-box {
        background-color: #fff9db !important;
        border: 1.5px solid #ffd43b !important;
        border-left: 7px solid #f59f00 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin: 12px 0px !important;
        color: #493200 !important;
        font-size: 0.96em !important;
        line-height: 1.5 !important;
    }
    /* Ubah kotak Expander menjadi putih bersih */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #b2dfdb !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #f0fdf4 !important;
        color: #004d40 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    /* Kotak kartu hasil prediksi */
    .result-card {
        background-color: #ffffff !important;
        padding: 22px;
        border-radius: 16px;
        border-left: 8px solid #0a9396;
        box-shadow: 0 6px 16px rgba(0, 95, 115, 0.12);
        margin-bottom: 20px;
    }
    /* Badge persentase biru toska dengan teks putih terang */
    .badge-score {
        background-color: #0077b6 !important;
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9em;
        display: inline-block;
    }
    /* Tombol hapus biasa */
    div[data-testid="stButton"] > button {
        background-color: #ffffff !important;
        color: #c92a2a !important;
        border: 1.5px solid #ffa8a8 !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    /* Warna link agar jelas terbaca */
    a {
        color: #0077b6 !important;
        font-weight: 600 !important;
        text-decoration: underline !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("♻️ EcoSort AI: Sistem Identifikasi & Pengelolaan Sampah")
st.write("Klik tombol **➕ Tambah / Pilih Sumber Gambar** di bawah untuk mengunggah foto, menempel gambar hasil *copy*, atau mengambil foto langsung dari kamera (**Maksimal 5 foto**).")

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
        "link_aksi_nama": "Drop Box E-Waste RJ / Dinas LH Kota",
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
        "link_aksi_nama": "Kitabisa.com (Donasi Pakaian) / Setali Indonesia",
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

# 4. Penyimpanan Sementara untuk Gambar Hasil Copy-Paste & Kamera
if "extra_images" not in st.session_state:
    st.session_state.extra_images = []
if "aktifkan_kamera" not in st.session_state:
    st.session_state.aktifkan_kamera = False

# 5. Menu Input Terpadu di Halaman yang Sama (Dalam Tanda +)
col_plus, col_reset, col_spacer = st.columns([1.8, 1.8, 2.8])

with col_plus:
    with st.popover("➕ Tambah / Pilih Sumber Gambar", use_container_width=True):
        st.markdown("### Pilih Cara Memasukkan Gambar:")
        
        # Opsi 1: Upload Gambar dari Perangkat
        st.markdown("**1. 📁 Upload Gambar dari File**")
        files_from_disk = st.file_uploader(
            "Pilih maksimal 5 file (JPG, JPEG, PNG)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="uploader_utama"
        )
        
        st.divider()
        
        # Opsi 2: Copy / Tempel Gambar dari Clipboard
        st.markdown("**2. 📋 Copy-Paste Gambar (Clipboard)**")
        st.caption("Klik kanan gambar di Google > *Copy image*, lalu klik tombol di bawah:")
        paste_result = paste_image_button(
            label="📋 Tempel Gambar Sekarang",
            background_color="#00897b",
            hover_background_color="#00695c",
            errors="ignore"
        )
        
        st.divider()
        
        # Opsi 3: Ambil Gambar Pakai Kamera
        st.markdown("**3. 📷 Ambil Foto dari Kamera**")
        if st.button("📷 Buka / Tutup Kamera", use_container_width=True):
            st.session_state.aktifkan_kamera = not st.session_state.aktifkan_kamera
            st.rerun()

with col_reset:
    if st.session_state.extra_images:
        if st.button("🗑️ Hapus Gambar Tempel / Kamera", use_container_width=True):
            st.session_state.extra_images = []
            st.rerun()

# Tampilkan Jendela Kamera di Halaman Utama jika Opsi Kamera Diaktifkan
if st.session_state.aktifkan_kamera:
    st.info("📷 **Mode Kamera Aktif:** Arahkan sampah ke kamera lalu klik tombol **Take Photo**.")
    cam_photo = st.camera_input("Ambil Foto Sampah")
    if cam_photo is not None:
        cam_bytes = cam_photo.getvalue()
        if not any(item["bytes"] == cam_bytes for item in st.session_state.extra_images):
            if len(st.session_state.extra_images) < 5:
                st.session_state.extra_images.append({
                    "image": Image.open(cam_photo).convert("RGB"),
                    "bytes": cam_bytes,
                    "name": f"Foto_Kamera_{len(st.session_state.extra_images)+1}.png"
                })
            else:
                st.warning("⚠️ Sudah mencapai batas maksimal 5 foto!")

# Simpan gambar hasil Copy-Paste ke daftar
if paste_result.image_data is not None:
    img_bytes = io.BytesIO()
    paste_result.image_data.save(img_bytes, format="PNG")
    img_val = img_bytes.getvalue()
    if not any(item["bytes"] == img_val for item in st.session_state.extra_images):
        if len(st.session_state.extra_images) < 5:
            st.session_state.extra_images.append({
                "image": paste_result.image_data.convert("RGB"),
                "bytes": img_val,
                "name": f"Gambar_Copy_{len(st.session_state.extra_images)+1}.png"
            })
        else:
            st.warning("⚠️ Sudah mencapai batas maksimal 5 foto!")

# Gabungkan Semua Gambar (dari Upload File + Copy-Paste + Kamera)
daftar_gambar = []
if files_from_disk:
    for f in files_from_disk:
        daftar_gambar.append({"image": Image.open(f).convert("RGB"), "name": f.name})
for item_extra in st.session_state.extra_images:
    daftar_gambar.append({"image": item_extra["image"], "name": item_extra["name"]})

# 6. Proses Prediksi & Tampilkan Top-3 Kemungkinan
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
            
            # Prediksi & Ambil 3 Kemungkinan Teratas (Top-3)
            predictions = model.predict(img_array)[0]
            top3_indices = np.argsort(predictions)[::-1][:3]
            
            top1_idx = top3_indices[0]
            top1_conf = predictions[top1_idx] * 100
            top1_key = CLASS_NAMES[top1_idx]
            detail1 = INFO_SAMPAH[top1_key]
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2.3])
            
            with col1:
                st.image(image, caption=f"Foto {idx+1}: {item
