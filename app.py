import streamlit as st
import pandas as pd
import joblib

# Judul aplikasi
st.title("🔍 Prediksi Kunjungan Ulang Pasien (30 Hari)")
st.markdown("Upload data pasien dan lihat apakah akan kembali dalam 30 hari ke depan berdasarkan model Random Forest (`rf7.pkl`).")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("rf7.pkl")

model = load_model()

# Upload file
uploaded_file = st.file_uploader("📁 Upload file data pasien (.csv)", type=["csv"])

if uploaded_file:
    # Baca data
    df = pd.read_csv(uploaded_file)

    # Tampilkan preview
    st.subheader("🧾 Data yang di-upload")
    # st.write(df.head())

    # Preprocessing (pastikan kolom seperti saat training)
    df['ADMISSION_DATE'] = pd.to_datetime(df['ADMISSION_DATE'], format='%d%m%Y', errors='coerce')
    df['ADMISSION_DAY'] = df['ADMISSION_DATE'].dt.dayofweek
    df['ADMISSION_MONTH'] = df['ADMISSION_DATE'].dt.month

    # Buat kolom target: apakah akan berkunjung lagi dalam 30 hari
    df['next_visit'] = df.groupby('MRN')['ADMISSION_DATE'].shift(-1)
    df['days_to_next'] = (df['next_visit'] - df['ADMISSION_DATE']).dt.days
    df['kunjungan_30_hari'] = df['days_to_next'].apply(lambda x: 1 if 0 < x <= 30 else 0)

    st.write(df.head())

    # fitur = df[['MRN', 'DPJP_CLEAN', 'ADMISSION_DAY', 'ADMISSION_MONTH']]
    # target = df['kunjungan_30_hari']
    # One-hot encoding
    # fitur_encoded = pd.get_dummies(fitur.astype(str))

    # Sesuaikan kolom agar match dengan model
    # model_columns = model.feature_names_in_
    # for col in model_columns:
    #     if col not in fitur_encoded.columns:
    #         fitur_encoded[col] = 0
    # fitur_encoded = fitur_encoded[model_columns]

    # Prediksi
    # prediksi = model.predict(fitur_encoded)
    # df['prediksi_kunjungan'] = prediksi

    # Tampilkan hasil
    # st.subheader("✅ Hasil Prediksi")
    # hasil = df[df['prediksi_kunjungan'] == 1][['MRN', 'NAMA_PASIEN', 'DPJP_CLEAN', 'ADMISSION_DATE']]
    # st.write(hasil)

    # Unduh hasil
    # hasil_csv = hasil.to_csv(index=False).encode('utf-8')
    # st.download_button("⬇️ Download Hasil Prediksi", hasil_csv, "hasil_prediksi.csv", "text/csv")

