import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

#(total images,2048)
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#2048 features will come with each image
#total images' vector will be plotted into 2048 dimension space
#the 5 vectors which will be closest to our new vector that'll be our recomendation

img = image.load_img('sample/stree.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# to calculate distance between featurelist and normalized result 
# we'll use nearest neighbors algorithm


#will find 6 recommandation
#brute as not that much data involved
#we tried euclidean here you guys can also try cosine
neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

#will return dst and indices
distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

#extract 0 item and then prints 1 2 3 4 5 6 which are rec
for file in indices[0][1:6]:

    #stores image in temp file 
    temp_img = cv2.imread(filenames[file])
    
    #shows the image
    cv2.imshow('output',cv2.resize(temp_img,(800,800)))
    
    #screen will stay
    cv2.waitKey(0)

