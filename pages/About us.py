import streamlit as st

st.title("About")

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

st.markdown('<p><h4>A personalized clothing recommendation system is a type of recommendation system that provides users with personalized recommendations about clothing in which they may be interested. The system attempts to exploit the knowledge graph for providing clothing recommendations to the user keeping the user context in mind. The recommendation is done by calculating the similarity in the clothing ontology similar to users collection</h4></p>', unsafe_allow_html=True)

