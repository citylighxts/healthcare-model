# 🏥 Aplikasi Prediksi Kategori Risiko Pasien

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://healthcare-model.streamlit.app/)

Sebuah aplikasi web *machine learning* sederhana yang dibangun menggunakan Scikit-learn dan Streamlit untuk memprediksi kategori risiko pasien (High, Medium, Low) berdasarkan data demografis dan rekam medis.

## 📜 Deskripsi & Konteks Proyek

Proyek ini adalah studi kasus *end-to-end* machine learning, mulai dari analisis data, *feature engineering*, pelatihan model, hingga *deployment* sebagai aplikasi web interaktif.

Tujuan utama dari aplikasi ini adalah untuk mengklasifikasikan pasien ke dalam salah satu dari tiga kategori risiko: **High Risk**, **Medium Risk**, atau **Low Risk**.

### ❗ Catatan Penting Mengenai Dataset

Dataset yang digunakan (`healthcare_dataset.csv`) adalah **data sintetis (buatan)**. Setelah Exploratory Data Analysis (EDA), ditemukan bahwa target asli (seperti `Test Results` atau `Medical Condition`) tidak memiliki korelasi statistik dengan fitur-fitur lainnya. Model yang dilatih untuk memprediksi target-target ini hanya menghasilkan akurasi yang sangat rendah (mendekati tebakan acak, ~27-40%).

**Solusi:**
Untuk membuktikan bahwa *pipeline* model berfungsi dengan benar, dilakukan *feature engineering* untuk **menciptakan target baru yang logis** bernama `Risk Category`. Target ini dibuat berdasarkan aturan bisnis (misal: "Pasien di atas 60 tahun dengan Diabetes" = "High Risk").

Model Random Forest kemudian dilatih untuk "belajar" dan menemukan kembali pola-pola ini. Hasilnya adalah akurasi ~100% pada data tes, yang membuktikan bahwa model berhasil mempelajari aturan yang konsisten dan logis.

## 🔗 Link Google Colab

Proses analisis, *feature engineering*, dan pelatihan model lengkap dapat dilihat di notebook Google Colab berikut:

* **[Healthcare Model Training Notebook](https://colab.research.google.com/drive/1gpHbYpylW2AyBtddIKqSWs6kBh8LNhGj?usp=sharing)**

## ✨ Fitur Aplikasi

* **Formulir Input Interaktif:** Pengguna dapat memasukkan data pasien melalui *sidebar* Streamlit.
* **Prediksi Real-time:** Model memberikan prediksi instan (`High Risk`, `Medium Risk`, `Low Risk`) setelah data di-submit.
* **Tampilan Probabilitas:** Menampilkan tingkat keyakinan (probabilitas) model untuk setiap kategori risiko.
* **Pipeline Lengkap:** Menggunakan *pipeline* Scikit-learn yang mencakup *pre-processing* (StandardScaler, OneHotEncoder) dan model (RandomForestClassifier) dalam satu file `.pkl`.

## 🛠️ Teknologi yang Digunakan

* **Analisis & Model:** Python, Pandas, Scikit-learn, Google Colab
* **Data Loading:** `gspread` (untuk membaca Google Sheets)
* **Aplikasi Web (Frontend):** Streamlit
* **Serialisasi Model:** `joblib`

## 📂 Struktur Proyek
├── 🚀 app.py # Kode inti aplikasi Streamlit 
├── requirements.txt # Daftar dependensi Python 
├── 📦 healthcare_model_rf.pkl # File pipeline model yang sudah dilatih 
├── 🏷️ label_encoder.pkl # File LabelEncoder untuk target 
└── 📄 README.md # File ini

## 🏃 Cara Menjalankan (Lokal)

1.  **Clone Repositori:**
    ```bash
    git clone [URL_REPOSITORI_ANDA]
    cd [NAMA_FOLDER_ANDA]
    ```

2.  **Buat dan Aktifkan Virtual Environment:**
    ```bash
    # Membuat venv
    python -m venv .venv

    # Aktivasi (macOS/Linux)
    source .venv/bin/activate

    # Aktivasi (Windows)
    .\.venv\Scripts\activate
    ```

3.  **Install Dependensi:**
    Pastikan file `requirements.txt` Anda memiliki versi `scikit-learn` yang **sama** dengan yang digunakan saat melatih model di Colab.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pastikan Model Ada:**
    Pastikan file `healthcare_model_rf.pkl` dan `label_encoder.pkl` (hasil download dari Colab) berada di folder yang sama dengan `app.py`.

5.  **Jalankan Aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

6.  Buka `http://localhost:8501` di browser Anda.