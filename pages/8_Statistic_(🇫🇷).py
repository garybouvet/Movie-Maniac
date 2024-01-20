
import streamlit as st
import fitz
from PIL import Image

# Définir les paramètres de configuration de la page
st.set_page_config(
    page_title="STATISTIC",
    layout="wide"
)

# Chemin vers votre fichier PDF
pdf_path = "./images/Analyse_Power_BI_Project_2.pdf"

# Textes personnalisés pour chaque image
custom_texts = [
    "Parmi le Top 100",
    "La place des femmes dans le cinéma",
    "L'histoire du cinéma",
    "Productions et budgets"
]

# Ouvrir le fichier PDF avec PyMuPDF
pdf = fitz.open(pdf_path)

# Parcourir les pages du PDF
for page_number in range(len(pdf)):
    # Extraire l'image de la page en tant que tableau Numpy
    page = pdf.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Afficher le texte personnalisé avec une taille de police augmentée et centrée
    st.markdown(f"## <center>{custom_texts[page_number]}</center>", unsafe_allow_html=True)

    # Afficher l'image dans Streamlit
    st.image(img)

