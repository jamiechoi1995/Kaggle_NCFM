from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

train_data_dir = '/media/wcai/8844289a-3afc-4ca9-98c0-a29abdb55c48/tdtd/wcai/ai_challenger_scene_train_20170904/scene_train_images_20170904'
val_data_dir = '/media/wcai/8844289a-3afc-4ca9-98c0-a29abdb55c48/tdtd/wcai/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'

learning_rate = 0.0001
SIZE = (299, 299)
nbr_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nbr_validation_samples = sum([len(files) for r, d, files in os.walk(val_data_dir)])

nbr_epochs = 25
batch_size = 32

Classes = []
for num in range(0,80):
        Classes.append(str(num))

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(80, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
# InceptionV3_model = load_model('weights.h5') #load model to continue your training
#InceptionV3_model.summary()

# optimizer=Adam(lr=0.001)
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = SIZE,
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = Classes,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size= SIZE,
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        #save_prefix = 'aug',
        classes = Classes,
        class_mode = 'categorical')

early_stopping = EarlyStopping(patience=5)

# autosave best Model
best_model_file = "./best_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

InceptionV3_model.fit_generator(
        train_generator,
        steps_per_epoch = nbr_train_samples/batch_size,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = nbr_validation_samples/batch_size,
        callbacks = [early_stopping, best_model])

InceptionV3_model.save('inceptv3_final.h5')