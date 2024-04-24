import streamlit as st
from PIL import Image



st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200&display=swap');

.big-font {
    font-size:90px !important;
    font-family: 'Poppins', sans-serif;
    font-weight: bolder;
}

.tool-headnig {
    margin-top: -20px;
    font-size:40px !important;
    font-family: 'Poppins', sans-serif;
    font-weight: bolder;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">My Wardrobe</p>' '<p><h3>Previous Searches: </h3></p>', unsafe_allow_html=True)

import os
import streamlit as st
from PIL import Image

# Path to the folder containing the images
#path = 'C:\\Users\\dyash\\OneDrive\\Desktop\\test\\fashion-recommender-system-main\\uploads'
path = 'uploads'
# path = '../uploads/'


# Iterate through the folder and display each image
for filename in os.listdir(path):
    img = Image.open(os.path.join(path, filename))
    st.image(img, caption=filename)