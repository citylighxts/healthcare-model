import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

st.set_page_config(page_title="Prediksi Kategori Risiko Pasien",
                   layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_model_and_encoder():
    try:
        model = joblib.load('healthcare_model_rf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError:
        st.error("Error: File model ('healthcare_model_rf.pkl' atau 'label_encoder.pkl') tidak ditemukan.")
        st.info("Pastikan Anda sudah menjalankan notebook Colab dan mengunduh kedua file .pkl ke folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

model, le = load_model_and_encoder()
le_classes = le.classes_

st.title('⚡ Aplikasi Prediksi Kategori Risiko Pasien')
st.markdown("""
Aplikasi ini memprediksi **Kategori Risiko** pasien (High, Medium, Low) 
berdasarkan data demografis, medis, dan kunjungan mereka.
""")
st.markdown("---")

st.sidebar.header('Masukkan Data Pasien:')

def user_input_features():
    st.sidebar.subheader("Data Diri & Klinis")
    age = st.sidebar.slider('Usia (Tahun)', 18, 100, 55)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    blood_type = st.sidebar.selectbox('Tipe Darah', ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'))
    medical_condition = st.sidebar.selectbox('Kondisi Medis', 
                                             ('Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Obesity', 'Cancer'))
    st.sidebar.subheader("Data Kunjungan & Obat")
    admission_type = st.sidebar.selectbox('Tipe Kunjungan', 
                                          ('Emergency', 'Urgent', 'Elective'))
    length_of_stay = st.sidebar.slider('Lama Menginap (Hari)', 1, 30, 5)
    medication = st.sidebar.selectbox('Obat-obatan', 
                                      ('Aspirin', 'Lipitor', 'Penicillin', 'Paracetamol', 'Ibuprofen'))
    st.sidebar.subheader("Administrasi")
    insurance_provider = st.sidebar.selectbox('Provider Asuransi', 
                                              ('Aetna', 'Blue Cross', 'Cigna', 'Unitedhealthcare', 'Medicare'))
    billing_amount = st.sidebar.number_input('Jumlah Tagihan ($)', min_value=100.0, max_value=50000.0, value=25000.0, step=100.0)
    data = {
        'Age': [age],
        'Gender': [gender],
        'Blood Type': [blood_type],
        'Medical Condition': [medical_condition],
        'Insurance Provider': [insurance_provider],
        'Billing Amount': [billing_amount],
        'Admission Type': [admission_type],
        'Medication': [medication],
        'Length of Stay': [length_of_stay]
    }
    feature_columns = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 
                       'Insurance Provider', 'Billing Amount', 'Admission Type', 
                       'Medication', 'Length of Stay']
    features = pd.DataFrame(data)
    features = features[feature_columns]
    return features

input_df = user_input_features()

st.subheader('Data Input Pasien:')
st.dataframe(input_df, use_container_width=True)
st.markdown("---")

if st.sidebar.button('🔮 Prediksi Kategori Risiko', use_container_width=True):
    try:
        prediction_encoded = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        prediction_text = le.inverse_transform(prediction_encoded)
        pred_prob_value = prediction_proba[0][prediction_encoded[0]]
        st.subheader('Hasil Prediksi:')
        result = prediction_text[0]
        if result == 'High Risk':
            st.error(f'**Prediksi: {result}**')
        elif result == 'Medium Risk':
            st.warning(f'**Prediksi: {result}**')
        else:
            st.success(f'**Prediksi: {result}**')
        st.metric(label="Tingkat Keyakinan (Probabilitas)", value=f"{pred_prob_value*100:.2f} %")
        st.subheader('Probabilitas Semua Kelas:')
        prob_df = pd.DataFrame({
            'Kategori Risiko': le_classes,
            'Probabilitas': [f"{p*100:.2f}%" for p in prediction_proba[0]]
        })
        st.dataframe(prob_df, use_container_width=True)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
else:
    st.info("Silakan masukkan data pasien di sidebar dan klik tombol 'Prediksi'.")
