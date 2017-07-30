import os
import csv
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
import sklearn
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam



from sklearn.model_selection import train_test_split


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_random_shadow(image):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = .15+.8*np.random.uniform()
    if np.random.randint(2)==1:
        #    random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def get_driving_log(path):
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    return lines

del_angle = 0.01
del_rate = 0.85

def generator(path, samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for image_nb in [0, 1, 2]:
                    name = path + 'IMG/'+ batch_sample[image_nb].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    image_b = random_brightness(image)
                    shift_dict = {0: 0, 1: 0.3, 2: -0.3}
                    angle = float(batch_sample[3]) + shift_dict.get(image_nb, "error")
                    if angle < del_angle: # to ignore zero steering angle data
                        if np.random.random() < del_rate:
                            continue
                


                    images.append(np.fliplr(image_b))
                    angles.append(-angle)
                
                    images.append(image_b)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# ------ Training/Validation data loading -------

path = './data/'

samples = get_driving_log(path)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(path, train_samples, batch_size=32)
validation_generator = generator(path, validation_samples, batch_size=32)

row, col, ch =160, 320, 3  # Trimmed image format
BATCH_SIZE = 32

def resize_im(x):
    from keras.backend import tf
    return tf.image.resize_images(x, (80, 160))

model = Sequential()
model.add(Cropping2D(cropping=((40,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1))
model.add(Lambda(resize_im))

model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only='true' , mode='auto')

model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_generator, samples_per_epoch = 10000, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5, callbacks=[checkpoint], verbose=1)
model.save('model.h5')

# Explicitly end tensorflow session
from keras import backend as K
    
K.clear_session()
