import pickle
import streamlit as st
import numpy as np

# Membaca model
diabetes_model = pickle.load(open('breast_model.sav', 'rb'))

# Judul web
st.title('Prediksi Diabetes')

# Input data dengan contoh angka valid untuk pengujian
mean_radius = st.text_input('mean_radius', '2')
mean_texture = st.text_input('mean_texture', '120')
mean_perimeter = st.text_input('mean_perimeter', '70')
mean_area = st.text_input('mean_area', '20')
mean_smoothness = st.text_input('mean_smoothness', '25.0')

Kanker_Payudara_diagnosis = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area),
                  float(mean_smoothness)]])
        


        # Lakukan prediksi
        diabetes_prediksi = diabetes_model.predict(inputs)
        
        if diabetes_prediksi[0] == 1:
            Kanker_Payudara_diagnosis = 'Tidak Terkena Kanker Payudara'
        else:
            Kanker_Payudara_diagnosis = 'Terkena Kanker Payudara'
            
        st.success(Kanker_Payudara_diagnosis)
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
