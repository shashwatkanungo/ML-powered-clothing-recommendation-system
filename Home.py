import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.set_page_config(
    page_title="SIX 5 SIX",
    page_icon="👋",
)


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

st.markdown('<p class="big-font">SIX5SIX</p>' '<p class="tool-headnig">ML powered Personalized Clothing Recommendation System</p>', unsafe_allow_html=True)


# st.title("Main Page")
st.sidebar.success("Select a page above.")

#if "my_input" not in st.session_state:
#   st.session_state["my_input"] = ""

#my_input = st.text_input("Input a text here", st.session_state["my_input"])
#submit = st.button("Submit")
#if submit:
#    st.session_state["my_input"] = my_input
#    st.write("You have entered: ", my_input)"""


#LOGIC STARTS

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")


