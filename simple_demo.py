import os
import sys
import random

import torch

from model import *
from utils import *

class Option():
  def __init__(self):
    self.nepoch = 400
    self.num_points = 2500
    self.model = '' # os.path.join('data','trained_models','svr_atlas_sphere.pth')
    self.cuda = False
    
opt = Option()

if opt.cuda and torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True
  opt.device = 'cuda'
else:
  opt.cuda = False
  opt.device = 'cpu'

with torch.no_grad():
	network = SVR_AtlasNet_SPHERE(num_points = opt.num_points).to(opt.device)
	network.apply(weights_init)
	
	if opt.model != '':
		if opt.cuda:
			network.load_state_dict(torch.load(opt.model))
		else:
			network.load_state_dict(torch.load(opt.model, map_location='cpu'))
		print("previous weight loaded")