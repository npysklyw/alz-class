import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Reshape,Dense,Conv2D,MaxPooling2D, Flatten,Input,Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import VGG16 
from tensorflow.python.keras.applications.vgg16 import preprocess_input



dim = (178,200)

zoom = [.99,1.01]

bright_range = [.8,1.2]

lr = 0.0001

batch = 32

eps = 30

momentum = .9
#brightness_range=bright_range,my
layers_unlocked = True
tf.keras.backend.get_session().run(tf.global_variables_initializer())
train_dr = ImageDataGenerator(rescale=1./255,fill_mode='constant',cval=0,
                                                           zoom_range=zoom,
                                                           data_format='channels_last',zca_whitening=False)

train_data_gen = train_dr.flow_from_directory(directory="AlzheimerDataset/train/",target_size=dim,
                                              batch_size=250)

test_dr = ImageDataGenerator(rescale=1./255,fill_mode='constant',cval=0,zoom_range=[1,1],
                                                          data_format='channels_last') 

test_data_gen = test_dr.flow_from_directory(directory="AlzheimerDataset/test/",target_size=dim,batch_size=250,
                                           shuffle = False)

train_data,train_labels =  train_data_gen.next()
test_data,test_labels = test_data_gen.next()

total_data = np.concatenate((train_data,test_data))
total_labels = np.concatenate((train_labels,test_labels))


#Get back the convolutional part of a VGG network trained on ImageNet
vg_model = VGG16(include_top=False,input_shape=(178,200,3),pooling='max')

vg_model.get_layer('block1_conv1').trainable = layers_unlocked
vg_model.get_layer('block1_conv2').trainable = layers_unlocked
vg_model.get_layer('block2_conv1').trainable = layers_unlocked
vg_model.get_layer('block2_conv2').trainable = layers_unlocked
vg_model.get_layer('block3_conv1').trainable = layers_unlocked
vg_model.get_layer('block3_conv2').trainable = layers_unlocked
vg_model.get_layer('block3_conv3').trainable = layers_unlocked
vg_model.get_layer('block4_conv1').trainable = layers_unlocked
vg_model.get_layer('block4_conv2').trainable = layers_unlocked
vg_model.get_layer('block4_conv3').trainable = layers_unlocked

## Add new trainable FC layers ##
flat = Flatten()(vg_model.output)
fc1 = Dense(1024,activation='relu', kernel_initializer='he_uniform')(flat) # put in kernel initializer he-uniform
dp1 = Dropout(0.25)(fc1)                                                   # changed dropout here from .5
output = Dense(4,activation='softmax')(dp1)                                # changed to sigmoid from softmax
vg_model = tf.keras.models.Model(inputs=vg_model.inputs, outputs=output)

vg_model.summary()

print(train_data.shape)

opt = SGD(lr=lr,momentum=0.9,nesterov=True)
vg_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=[ 'accuracy'])

model_history = vg_model.fit(
    train_data, train_labels,
    epochs=eps,batch_size=batch,shuffle=True,validation_data=(test_data,test_labels)
    )
scores = vg_model.evaluate(train_data, train_labels)
tscores = vg_model.evaluate(test_data, test_labels)
print("Accuracy: %.2f%%" %(scores[1]*100) + " TAccuracy: %.2f%%" %(tscores[1]*100))
vg_model.save_weights('third_try_vgg16.h5')



