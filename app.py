#here we'll import libraries
#for all images we've to generate embedding
#embedding= 2048 set of numbers which will represent our images
#we'll use ResNet model by using this model we'll generate embedding


import tensorflow

#importing image for methods to load, prepare and process images. 
from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm

#images are in this directory
import os

#for loop's progress rep
from tqdm import tqdm
import pickle


#calling ResNet class and pass parameters 
#224,224,3 is standard res for an image in model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) 
model.trainable = False


#we're not training our model. by using imagenet we're predicting. 
#that prediction is also not end to end cuz we're removing top layer and including our own top layer
#sending model in sequential. new top layer globalmaxpooling2d
# we used ResNet module till "model,"

#seq=each layer has exactly one input and output and is stacked together to form the entire network
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#for debugging
#print(model.summary())


#
def extract_features(img_path,model):

    #loads image in env
    img = image.load_img(img_path,target_size=(224,224))
    
    #converts img into numpy array (3d) 224,224,3
    img_array = image.img_to_array(img)
    
    #image array passes through expand dims and resized will be 1,224,224,3 
    #reps a batch of images in 1 image(4d)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    #converts img to rgb to bgr as resnet module will only accept that input only
    preprocessed_img = preprocess_input(expanded_img_array)
    
    #sending proccesed img here
    #then it'll give shape of (1,2048) by model.predict.shape function
    #flatten will convert 2d shape to 1d
    result = model.predict(preprocessed_img).flatten()
    
    #value between 0 n 1
    #divides by 313 point something
    normalized_result = result / norm(result)

    return normalized_result

#empty intiially
filenames = []

#for loop for all files in os directory
for file in os.listdir('images'): 
    filenames.append(os.path.join('images',file))

#2d list of 2048 features for each image
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

#it'll export in file
#wb=write binary
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

