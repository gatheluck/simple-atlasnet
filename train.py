from dataset import *
from options import *
from model import *
from utils import *

import torch


opt = TrainOptions().parse()

dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.class_choice, train=opt.use_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))

print('training set: ', len(dataset.datapath))

model = SVR_AtlasNet_SPHERE(num_points = opt.num_points)
model.apply(weights_init)
model = model.to(opt.device)