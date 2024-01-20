import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MOVIE MANIAC",
    layout="wide"
)

logo_path = "./images/small_movie_maniac.png"  

# Load the logo image
logo_image = Image.open(logo_path)

# Create 3 columns with the middle column being wider to center the logo
col1, col2, col3 = st.columns([1, 6, 1])

# Use col2 to center the logo
with col2:
    st.image(logo_image, use_column_width=True)