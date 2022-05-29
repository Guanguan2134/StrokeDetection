import argparse
import time
import glob
import shutil
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import UNet, ResUNet
from data import trainGenerator, testGenerator, make_val, saveResult
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_dir", default="data", type=str, help="path to dataset")
parser.add_argument("-o", "--output_dir", default="data/results", type=str, help="path to testing output")
parser.add_argument("-m", "--model", default="UNet", type=str, help="UNet or ResUNet")
parser.add_argument("-a", "--activation", default="relu", type=str, help="Activation function of UNet, ex: relu, selu, elu, leaky_relu")
parser.add_argument("-mp", "--model_path", type=str, help="path to output model")
parser.add_argument("-e", "--epoch", default=10, type=int, help="how many epochs to train")
parser.add_argument("-s", "--step_per_epoch", default=500, type=int, help="step per epoch")
parser.add_argument("-v", "--val_split", default=0.2, type=int, help="validation split ratio of training dataset")
args = parser.parse_args()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Data generator
if os.path.exists(os.path.join("data", "val")):
    imgs = glob.glob(os.path.join("data", "val", "image", "*"))
    labels = glob.glob(os.path.join("data", "val", "label", "*"))
    for img, lab in zip(imgs, labels):
        shutil.move(img, img.replace("val", "train"))
        shutil.move(lab, lab.replace("val", "train"))

train_img_path = glob.glob(os.path.join(args.dataset_dir, "train", "image", "*.png"))
val_size = int(round(len(train_img_path)*args.val_split))
_, val_path = train_test_split(train_img_path, test_size=val_size)
make_val(dataset_dir=args.dataset_dir, val_path=val_path)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
train_gen = trainGenerator(batch_size=5, train_path='data/train', 
                        image_folder='image', mask_folder='label', aug_dict=data_gen_args)
val_gen = trainGenerator(batch_size=5, train_path='data/val', 
                        image_folder='image', mask_folder='label', aug_dict=data_gen_args)
test_gen = testGenerator("data/test/image", mask_path="data/test/label")

# Train model
print(f"[INFO] Train on {args.model} model")
model = None
if args.model == "UNet":
    exec(f"model = {args.model}(activation='{args.activation.lower()}')")
    activation = args.activation.lower()
else:
    exec(f"model = {args.model}()")
    activation = ""

if args.model_path is None:
    model_path = os.path.join("model", "_".join(["strokeSeg", time.strftime("%m%d", time.localtime()), args.model.lower(), activation.lower()]).strip("_")+".h5")
else:
    model_path = args.model_path
print(f"[INFO] The model will be saved to {model_path}")

history = model.fit(train_gen, steps_per_epoch=args.step_per_epoch, 
                    validation_data=val_gen, validation_steps=10,
                    epochs=args.epoch,
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
                               ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=0.0001, cooldown=0, min_lr=0, verbose=1)])
for path in val_path:
    shutil.move(os.path.join(args.dataset_dir, "val", os.path.basename(os.path.dirname(path)), os.path.basename(path)), path)
    path = path.replace(os.path.basename(os.path.dirname(path)), "label")
    shutil.move(os.path.join(args.dataset_dir, "val", os.path.basename(os.path.dirname(path)), os.path.basename(path)), path)
shutil.rmtree(os.path.join(args.dataset_dir, "val"))

# Plot and generate results
acc = history.history['dice_coef'] if args.model == 'ResUNet' else history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_dice_coef'] if args.model == 'ResUNet' else history.history['val_accuracy']
val_loss = history.history['val_loss']
epoch = [i for i in range(len(acc))]

plt.figure(figsize=(25,10))
plt.subplot(121)
plt.plot(epoch, acc, label='Train')
plt.plot(epoch, val_acc, label='Val')
plt.grid('on')
plt.xlabel('Epochs')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epoch, loss, label='Train')
plt.plot(epoch, val_loss, label='Val')
plt.grid('on')
plt.xlabel('Epochs')
plt.title('Training and validation loss')
plt.legend()

plt.savefig("fig/train_metric.png")

results = model.predict(test_gen, 14, verbose=1)
saveResult(args.output_dir, results)
print(f"[INFO] The testing results are saved to {args.output_dir}")
