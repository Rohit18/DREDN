#Import libraries
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

#Define paths to the training sets. path_l - MODIS simulated 1000m low res (for optical bands), path _s = MODIS 500m original at a scale factor of 2x
path_s = '/users/PAS1437/osu10674/MM/Datasets/M3_HR'
path_l = '/users/PAS1437/osu10674/MM/Datasets/M3_LR'

#Default learning rate value, update after using the LR Finder to find the optimal learning rate
lr = 1e-4

#Method for training the model
#Set the number of epochs
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(90, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results()

#Load data
#Set the transforms
#Remember, this is for the higher resolution dataset
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_s + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs))
    return data

#Set Batch Size
#Set Image Size
#Set architecture
#Set the ImageList (Custom or Default?)
#Set the Validation Default
#Set normalization
#Set loss
#Set pretrained or from_scratch
#Set Weight Decay

bs,size= 6, 512
arch = models.resnet34
src = ImageImageList.from_folder(path_l).split_by_rand_pct(0.01, seed=42)
data = get_data(bs,size)
wd = 1e-3

learn = unet_learner(data, arch, wd=wd, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight, self_attention=True, metrics=[root_mean_squared_error], pretrained=True)

src = ImageImageList.from_folder(path_s).split_none()

#train only the final layers, name the model
do_fit('tmp_M3_PT')

#Unfreeze the weights of the other layers and train the entire network
learn.unfreeze()
do_fit('M3_PT')

#Location for generating results
path_gen = '/MM/Datasets/M3_PT_Valid/'

#For saving model predictions from the Validation Set
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen + '/' + names[i].name)
            i += 1

#valid_dl for validation and fix_dl for entire dataset
save_preds(data.valid_dl)