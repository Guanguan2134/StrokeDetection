from data import saveResult, testGenerator, trainGenerator
from tensorflow.keras.models import load_model
from evaluation_matrix import evaluation
from model import exp_dice_loss, dice_coef
import os, glob
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-mp", "--model_path", type=str, default="model/strokeseg_0529_unet_leaky_relu.h5", help="path to output stroke segmentation detector model")
args = parser.parse_args()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
test_gen = testGenerator("data/test/image", mask_path="data/test/label")
train_gen = trainGenerator(batch_size=1, train_path='data/train', 
                        image_folder='image', mask_folder='label', aug_dict=data_gen_args)

model = load_model(args.model_path, custom_objects={ 'loss': exp_dice_loss, 'dice_coef': dice_coef }, compile=False)

results = model.predict(test_gen)
saveResult("data/results", results)

# test_gen = testGenerator("data/test/image", mask_path="data/test/label")
# test_acc, test_loss = model.evaluate(test_gen)
# print(f"test acc: {test_acc*100:.2f}%. test loss: {test_loss:.4f}")

dice, sen, spe, IOU = [], [], [], []
test_gen = testGenerator("data/test/image", mask_path="data/test/label")
for i in range(len(glob.glob(os.path.join("data", "test", "image", "*.png")))):
    (img, mask) = next(test_gen)
    dices, sens, spes, ious = evaluation(model, img, mask[0], verbose=0)
    dice.append(dices)
    sen.append(sens)
    spe.append(spes)
    IOU.append(ious)
Dice = np.mean(dice)
Sen = np.mean(sen)
Spe = np.mean(spe)
IoU = np.mean(IOU)

print()
print("============================")
print("Dice score : {:0.4f}".format(Dice))
print("Sensitivity score : {:0.4f}".format(Sen))
print("Specificity score : {:0.4f}".format(Spe))
print("IoU : {:0.4f}".format(IoU))
print("============================")