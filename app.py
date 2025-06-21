import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load model dan scaler
model = joblib.load('C:/Users/Shabrina/Storage/University/Semester 6/Data Challenge/Mitra UAS/model_unitA.pkl')
scaler_X = joblib.load('C:/Users/Shabrina/Storage/University/Semester 6/Data Challenge/Mitra UAS/scaler_X.pkl')
scaler_y = joblib.load('C:/Users/Shabrina/Storage/University/Semester 6/Data Challenge/Mitra UAS/scaler_y.pkl')

st.title("🧴 Prediksi Unit A yang Terjual")
st.write("Pilih tanggal untuk memprediksi jumlah Unit A yang terjual.")

# Input tanggal
tanggal = st.date_input("Pilih Tanggal", datetime.today())

# Tentukan nilai Day otomatis berdasarkan tanggal yang dipilih
if tanggal.weekday() >= 5:
    default_day = 1  # Weekend
else:
    default_day = 0  # Weekday

# Nilai default fitur lain (misal dari data training, bisa disesuaikan)
default_grp_a = 2.47
default_grp_b = 0.07
default_unit_b = 59.7
default_toko1 = 1
default_toko3 = 1
default_toko4 = 1
default_toko5 = 1
default_toko6 = 0

if st.button("Prediksi"):
    # Buat DataFrame input
    input_df = pd.DataFrame([{
        'GRP_A_adstock': default_grp_a,
        'GRP_B_adstock': default_grp_b,
        'Unit_B': default_unit_b,
        'Day': default_day,
        'Toko1': default_toko1,
        'Toko3': default_toko3,
        'Toko4': default_toko4,
        'Toko5': default_toko5,
        'Toko6': default_toko6
    }])

    # Scaling
    input_scaled = scaler_X.transform(input_df)

    # Prediksi
    pred_scaled = model.predict(input_scaled)
    pred_unit = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    # Hitung CI 80%
    resid_std = 26.32  # ganti dengan RMSE dari model training
    z_score = 1.28  # z untuk 80% confidence
    ci_lower = pred_unit - z_score * resid_std
    ci_upper = pred_unit + z_score * resid_std

    st.success(
        f"Prediksi Unit A terjual pada {tanggal} adalah **{pred_unit:.0f} unit**\n\n"
        f"Selang kepercayaan 80%: **{ci_lower:.0f} - {ci_upper:.0f} unit**"
    )
