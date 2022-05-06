from model import *
from SeLUNet_model import *
from ResUNet_model import *
from data import *
from evaluation_matrix import *
import tensorflow as tf
from PIL import Image

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
myGene = trainGenerator(5, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
testGene = testGenerator("data/membrane/test")

model = unet3()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='exp_dice_loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=500, epochs=10, callbacks=[model_checkpoint])

# model = unet3()
# model.load_weights('unet_membrane.hdf5')

results = model.predict_generator(testGene, 15, verbose=1)
saveResult("data/membrane/results", results)
