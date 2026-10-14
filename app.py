import streamlit as st
import pandas as pd
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Pelestari Bahasa Nusantara - Patriotisme Digital",
    page_icon="🇮🇩",
    layout="centered"
)

# Custom CSS Styling untuk UI yang profesional & estetis
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #dc2626;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #b91c1c;
        box-shadow: 0 6px 8px -1px rgba(185, 28, 28, 0.3);
    }
    .card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .badge {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Memuat Dataset Korpus NusaX dan Model AI
@st.cache_resource
def load_system():
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
    
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return corpus_dict, tokenizer, model

with st.spinner("🔄 Memuat sistem kecerdasan buatan dan pangkalan data korpus NusaX..."):
    corpus_dict, tokenizer, model = load_system()

# Sidebar Informasi & Panduan
with st.sidebar:
    st.markdown("### 🇮🇩 Tentang Proyek")
    st.markdown("""
    **Pelestari Bahasa Nusantara** adalah wujud nyata **Patriotisme Digital** dalam menjaga kebhinekaan bahasa daerah Indonesia di ruang siber.
    
    * **Metodologi:** Hibrida (*Corpus Matching* + *AI NLLB*)
    * **Sumber Korpus:** NusaX (IndoNLP)
    * **Mata Kuliah:** Pendidikan Patriotisme
    * **Pengembang:** Teknik Informatika - UNTAG Surabaya
    """)
    st.markdown("---")
    st.markdown("💡 **Contoh Kalimat Pengujian:**")
    if st.button("Cinta tanah air selamanya"):
        st.session_state["preset_text"] = "Cinta tanah air selamanya"
    if st.button("Jaga persatuan bangsa kita"):
        st.session_state["preset_text"] = "Jaga persatuan bangsa kita"
    if st.button("Kibarkan sang saka merah putih"):
        st.session_state["preset_text"] = "Kibarkan sang saka merah putih"

# Header Utama
st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
st.markdown("<span class='badge'>Pendidikan Patriotisme & Informatika</span>", unsafe_allow_html=True)
st.markdown("<h1 style='color: #1e293b; margin-top: 0.5rem;'>Pelestari Bahasa Nusantara</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.05rem;'>Merawat Kebhinekaan Indonesia Melalui Teknologi AI & Korpus Lokal</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Container Utama (Card Layout)
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    default_text = st.session_state.get("preset_text", "")
    
    text_input = st.text_area(
        "**Teks (Bahasa Indonesia):**", 
        value=default_text,
        placeholder="Contoh: Mari kita jaga persatuan dan kesatuan bangsa...",
        height=100
    )
    
    target_lang = st.selectbox(
        "**Pilih Bahasa Daerah Tujuan:**",
        options=["jav_Latn", "sun_Latn", "min_Latn"],
        format_func=lambda x: {
            "jav_Latn": "🟡 Bahasa Jawa",
            "sun_Latn": "🔵 Bahasa Sunda",
            "min_Latn": "🟢 Bahasa Minangkabau (Padang)"
        }[x]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        translate_btn = st.button("Terjemahkan Sekarang", use_container_width=True)
    
    if translate_btn:
        if not text_input.strip():
            st.warning("⚠️ Mohon masukkan teks Bahasa Indonesia terlebih dahulu!")
        else:
            with st.spinner("🤖 Sistem sedang memproses terjemahan..."):
                clean_query = text_input.lower().strip()
                translated_text = ""
                source = ""
                
                # Prioritas 1: Korpus NusaX
                if clean_query in corpus_dict:
                    translated_text = corpus_dict[clean_query].get(target_lang, "")
                    if translated_text and translated_text != "nan":
                        source = "NusaX Official Corpus (IndoNLP)"
                
                # Prioritas 2: Model AI NLLB
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
                
                if translated_text:
                    st.markdown("---")
                    st.markdown("### 🎯 Hasil Terjemahan:")
                    st.success(translated_text)
                    st.caption(f"📌 Sumber Rujukan Sistem: **{source}**")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>© 2026 • Proyek Akhir Pendidikan Patriotisme & Teknik Informatika • Universitas 17 Agustus 1945 Surabaya</p>", unsafe_allow_html=True)
