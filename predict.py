from keras.models import load_model
import os
import json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000

weights_path = ('./weights.h5')

#test_data_dir = ('/media/wcai/8844289a-3afc-4ca9-98c0-a29abdb55c48/tdtd/wcai/ai_challenger_scene_test_a_20170922')
test_data_dir = ('/home/wcai/Desktop/0/')

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

test_image_list = test_generator.filenames

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

result=[]
for i, image_name in enumerate(test_image_list):
    temp_dict = {}
    index=[]
    prediction = sorted(enumerate(predictions[i]), key=lambda x:x[1], reverse=True)[:3]
    for x, y in prediction:
        index.append(x)
        temp_dict['image_id'] = image_name.split('/')[1]
        temp_dict['label_id'] = index
    result.append(temp_dict)

with open('submit.json', 'w') as f:
    json.dump(result, f)

print('Submission file successfully generated!')
