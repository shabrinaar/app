import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load model dan scaler
model = joblib.load('model_unitA.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

st.title("📦 Prediksi Unit A yang Terjual")
st.write("Pilih tanggal untuk memprediksi jumlah Unit A yang terjual.")

# Input tanggal
tanggal = st.date_input("Pilih Tanggal", datetime.today())

# Nilai default fitur lain (misal dari data training, bisa disesuaikan)
default_grp_a = 2.47
default_grp_b = 0.07
default_unit_b = 59.7
default_toko1 = 1
default_toko3 = 1
default_toko4 = 1
default_toko5 = 1
default_toko6 = 0
default_day = 0

if st.button("Prediksi"):
    # Buat DataFrame dengan urutan kolom sesuai saat training
    input_df = pd.DataFrame([{
        'Unit_B': default_unit_b,
        'GRP_A_adstock': default_grp_a,
        'GRP_B_adstock': default_grp_b,
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


    st.success(f"Prediksi Unit A terjual pada {tanggal} adalah **{pred_unit:.0f} unit**")
