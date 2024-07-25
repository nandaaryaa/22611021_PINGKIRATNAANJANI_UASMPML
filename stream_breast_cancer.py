import pickle
import streamlit as st
import numpy as np

# Membaca model
breast_cancer_model = pickle.load(open('breast_model.sav', 'rb'))

# Judul web
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 1em;
            font-weight: bold;
        }
        .input-container {
            margin-bottom: 1em;
        }
        .stTextInput label {
            font-size: 1.1em;
            color: #34495E;
            font-weight: bold;
        }
        .stTextInput input {
            background-color: #E8F8F5;
            color: #1A5276;
            font-size: 1.1em;
        }
        .submit-button {
            display: flex;
            justify-content: center;
            margin-top: 2em;
        }
        .stButton button {
            background-color: #28B463;
            color: white;
            width: 200px;
            height: 50px;
            font-size: 1.2em;
            font-weight: bold;
            border-radius: 10px;
        }
        .result-success {
            color: #28B463;
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            margin-top: 1em;
        }
        .result-error {
            color: #E74C3C;
            font-size: 1.2em;
            text-align: center;
            margin-top: 1em;
        }
    </style>
""", unsafe_allow_html=True)


# Form input
with st.form(key='prediction_form'):
    st.markdown('<div class="title">Prediksi Kanker Payudara</div>', unsafe_allow_html=True)
    mean_radius = st.text_input('Mean Radius', '19.81', help='Contoh: 19.81')
    mean_texture = st.text_input('Mean Texture', '22.15', help='Contoh: 22.15')
    mean_perimeter = st.text_input('Mean Perimeter', '130', help='Contoh: 130')
    mean_area = st.text_input('Mean Area', '1260', help='Contoh: 1260')
    mean_smoothness = st.text_input('Mean Smoothness', '0.09831', help='Contoh: 0.09831')
    
    # Centered submit button
    st.markdown('<div class="submit-button">', unsafe_allow_html=True)
    submit_button = st.form_submit_button(label='Prediksi')
    st.markdown('</div>', unsafe_allow_html=True)

Kanker_Payudara_diagnosis = ''

# Membuat tombol untuk prediksi
if submit_button:
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area),
                  float(mean_smoothness)]])
        
        # Lakukan prediksi
        kanker_payudara_prediksi = breast_cancer_model.predict(inputs)
        
        if kanker_payudara_prediksi[0] == 1:
            Kanker_Payudara_diagnosis = 'Pasien Terkena Kanker Payudara'
            st.markdown(f'<div class="result-success">{Kanker_Payudara_diagnosis}</div>', unsafe_allow_html=True)
        else:
            Kanker_Payudara_diagnosis = 'Pasien Tidak Terkena Kanker Payudara'
            st.markdown(f'<div class="result-success">{Kanker_Payudara_diagnosis}</div>', unsafe_allow_html=True)
    except ValueError:
        st.markdown('<div class="result-error">Pastikan semua input diisi dengan angka yang valid.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="result-error">Terjadi kesalahan: {e}</div>', unsafe_allow_html=True)
