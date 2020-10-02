import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

path_hr = '/users/PAS1437/osu10674/MM/Datasets/M31_HR'
path_lr = '/users/PAS1437/osu10674/MM/Datasets/M31_LR'

lr = 1e-4

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(30, lrs, pct_start=pct_start)
    learn.save(save_name)

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr + '/' + x.name)
           .transform(get_transforms(do_flip=True), size=size, tfm_y=True)
           .databunch(bs=bs))
    return data

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

bs,size = 6,512
arch = models.resnet34

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.01, seed=42)

data = get_data(bs,size)
t = data.valid_ds[0][1].data
t = torch.stack([t,t])
gram_matrix(t)
base_loss = F.l1_loss
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]

feat_loss = FeatureLoss(vgg_m, blocks[:3], [5,15,2])

wd = 1e-3

learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics, blur=True, norm_type=NormType.Weight, self_attention=True, metrics=[root_mean_squared_error])

do_fit('tmp_M31_FL')

learn.unfreeze()

do_fit('M31_FL')

#Where you want to store the results and compare
path_gen = '/users/PAS1437/osu10674/MM/Datasets/M31_FL_Valid/'

#For saving model predictions
#Check the output format of the predictions!
def save_preds(dl):
    i=0
    names = dl.dataset.items
    
    for b in dl:
        preds = learn.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen + '/' + names[i].name)
            i += 1

save_preds(data.valid_dl)