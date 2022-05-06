from model import *
from SeLUNet_model import *
from ResUNet_model import *
from data import *
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(5,'data/membrane/train_old','image','label',data_gen_args,save_to_dir = None)
model = unet3()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=500,epochs=10,callbacks=[model_checkpoint])

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
epoch = [i for i in range(len(acc))]

plt.figure()
plt.plot(epoch, acc, label='Accuracy')
plt.plot(epoch, loss, label='Loss')
plt.grid('on')
plt.xlabel('Epochs')
plt.title('Training accuracy and loss')
plt.legend()
plt.show()