import streamlit as st
import pandas as pd
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Pelestari Bahasa Nusantara",
    page_icon="🇮🇩",
    layout="centered"
)

# Memuat Dataset Korpus NusaX dan Model AI (Menggunakan cache agar cepat)
@st.cache_resource
def load_system():
    # 1. Muat Korpus NusaX dari train.csv
    corpus_dict = {}
    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv")
        for _, row in df.iterrows():
            indo_text = str(row.get('indonesian', '')).lower().strip()
            if indo_text and indo_text != 'nan':
                corpus_dict[indo_text] = {
                    "jav_Latn": str(row.get('javanese', '')),
                    "sun_Latn": str(row.get('sundanese', '')),
                    "min_Latn": str(row.get('minangkabau', ''))
                }
    
    # 2. Muat Model AI NLLB
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return corpus_dict, tokenizer, model

with st.spinner("Sedang memuat sistem AI dan korpus NusaX ke server... Mohon tunggu sebentar."):
    corpus_dict, tokenizer, model = load_system()

# Tampilan Antarmuka (UI)
st.markdown("<h1 style='text-align: center; color: #b91c1c;'>🇮🇩 Pelestari Bahasa Nusantara</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Merawat Kebhinekaan Indonesia Melalui Teknologi Kecerdasan Buatan & Korpus NusaX</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Teks
text_input = st.text_area("Masukkan Teks (Bahasa Indonesia):", placeholder="Contoh: Cinta tanah air selamanya...")

# Pilihan Bahasa
target_lang = st.selectbox(
    "Pilih Bahasa Daerah Tujuan:",
    options=["jav_Latn", "sun_Latn", "min_Latn"],
    format_func=lambda x: {
        "jav_Latn": "Bahasa Jawa",
        "sun_Latn": "Bahasa Sunda",
        "min_Latn": "Bahasa Minangkabau"
    }[x]
)

# Tombol Terjemah
if st.button("Terjemahkan Sekarang", use_container_width=True):
    if not text_input.strip():
        st.warning("Mohon masukkan teks Bahasa Indonesia terlebih dahulu!")
    else:
        clean_query = text_input.lower().strip()
        translated_text = ""
        source = ""
        
        # Prioritas 1: Cek Korpus NusaX
        if clean_query in corpus_dict:
            translated_text = corpus_dict[clean_query].get(target_lang, "")
            if translated_text and translated_text != "nan":
                source = "NusaX Official Corpus (IndoNLP)"
        
        # Prioritas 2: Fallback ke Model AI NLLB
        if not translated_text or translated_text == "nan":
            try:
                tokenizer.src_lang = "ind_Latn"
                inputs = tokenizer(text_input, return_tensors="pt")
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                    max_length=150
                )
                translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                source = "AI Pre-trained Model (NLLB-200)"
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model AI: {e}")
        
        # Tampilkan Hasil
        if translated_text:
            st.success("Terjemahan Berhasil!")
            st.markdown(f"**Hasil Terjemahan:**")
            st.info(translated_text)
            st.caption(f"Sumber Rujukan Sistem: `{source}`")