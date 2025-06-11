import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# Load model dan alat bantu
# === Load model dan data training ===
model = joblib.load('model_svm.pkl')
df_final = joblib.load('df_final.pkl')

st.title("=== INPUT DATA PASIEN ===")

no = st.text_input("No:")
nama = st.text_input("Nama:")
umur = st.number_input("Umur", min_value=0)
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
demam = st.selectbox("Demam", ['YA', 'TIDAK'])
pendarahan = st.selectbox("Pendarahan", ['YA', 'TIDAK'])
pusing = st.selectbox("Pusing", ['YA', 'TIDAK'])
nyeri_otot_sendi = st.selectbox("Nyeri Otot/Sendi", ['YA', 'TIDAK'])
trombosit = st.number_input("Trombosit (x1000)", min_value=0.0, format="%.3f")
hemoglobin = st.number_input("Hemoglobin", min_value=0.0, format="%.2f")
hematokrit = st.number_input("Hematokrit", min_value=0.0, format="%.2f")

if st.button("Prediksi"):
    # Mapping string ke numerik
    map_ya_tidak = {'YA':1, 'TIDAK':0}
    map_jk = {'Laki-laki':1, 'Perempuan':2}

    data_dict = {
        'NO': [int(no) if no.isdigit() else 0],
        'Umur': [umur],
        'Demam': [map_ya_tidak[demam]],
        'Pendarahan': [map_ya_tidak[pendarahan]],
        'Pusing': [map_ya_tidak[pusing]],
        'Nyeri Otot/Sendi': [map_ya_tidak[nyeri_otot_sendi]],
        'Trombosit': [trombosit],
        'Hemoglobin': [hemoglobin],
        'Hematokrit': [hematokrit],
        'Jenis_kelamin': [map_jk[jenis_kelamin]]
    }

    df_input = pd.DataFrame(data_dict)

    st.write("=== [DEBUG] Data sebelum scaling:")
    st.dataframe(df_input)

    # Reindex sesuai kolom fit_columns model
    df_input = df_input.reindex(columns=fit_columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(df_input)

    # Prediksi
    hasil_prediksi = model.predict(input_scaled)[0]

    label_map_output = {
        1: "DD (Demam Dengue)",
        2: "DBD (Demam Berdarah Dengue)",
        3: "DSS (Dengue Shock Syndrome)"
    }
    st.write("=== HASIL PREDIKSI ===")
    st.write(f"Nama: {nama}")
    st.success(f"Prediksi: {label_map_output.get(hasil_prediksi, 'Tidak diketahui')}")
