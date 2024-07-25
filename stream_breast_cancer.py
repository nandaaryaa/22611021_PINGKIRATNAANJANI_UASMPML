import pickle
import streamlit as st
import numpy as np

# Membaca model
breast_cancer_model = pickle.load(open('breast_model.sav', 'rb'))

# Judul web
st.title('Prediksi Kanker Payudara')

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 2.5em;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1em;
    }
    .input-container {
        margin-bottom: 1em;
    }
    .stTextInput label {
        font-size: 1.1em;
        color: #34495E;
    }
    .stTextInput input {
        background-color: #E8F8F5;
        color: #1A5276;
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
    }
</style>
""", unsafe_allow_html=True)

# Form input
with st.form(key='prediction_form'):
    mean_radius = st.text_input('Mean Radius',)
    mean_texture = st.text_input('Mean Texture', )
    mean_perimeter = st.text_input('Mean Perimeter', )
    mean_area = st.text_input('Mean Area', )
    mean_smoothness = st.text_input('Mean Smoothness', )
    
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
            Kanker_Payudara_diagnosis = ' Terkena Kanker Payudara'
        else:
            Kanker_Payudara_diagnosis = 'Tidak Terkena Kanker Payudara'
        st.success(Kanker_Payudara_diagnosis)
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
