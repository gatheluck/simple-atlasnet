from dataset import *
from options import *
from model import *
from utils import *

import os
import sys

import numpy as np
import scipy.io as sio

import torch
import torch.optim as optim

import meshzoo

try:
	import visdom
	vis = visdom.Visdom()
except ImportError as err:
	vis = None
	print("visdom is not available")

try:
	import soft_renderer as sr
except ImportError as err:
	sr = None
	print("soft_renderer is not available")

if __name__ == '__main__':

	opt = TestOptions().parse()
	# vis = visdom.Visdom(port = 8888)

	dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.class_choice, train=opt.use_train)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))
	len_dataset = len(dataset)
	print('test set: ', len(dataset.datapath))

	model = SVR_AtlasNet_SPHERE()
	model.load_state_dict(torch.load(opt.weight))
	print("previous weight loaded")
	model = model.to(opt.device)

	# a = sio.loadmat(os.path.join('data','points_sphere.mat'))
	# points_sphere = np.array(a['p'])
	# points_sphere = torch.FloatTensor(points_sphere).transpose(0,1).contiguous() 
	# print(points_sphere.shape) # torch.Size([3, 7446])
	
	n_subdivide=4
	
	# inputs of sr.Mesh class should be like following. 
	# verts: (#batch, #vertices, 3)
	# faces: (#batch, #faces, 3)

	verts, faces = [elem.transpose(0,1)[np.newaxis, ...] for elem in meshzoo.iso_sphere(n_subdivide)]
	
	# print(verts.shape) # (1, 2562, 3)
	# print(faces.shape) # (1, 5120, 3)

	# logger
	test_loss = AverageValueMeter()
	test_curve = []

	test_loss.reset()
	model.eval()

	with torch.no_grad():
		for i, data in enumerate(dataloader, 0):  
			img, points, label, _ , _= data
			img, points = img.to(opt.device), points.to(opt.device)

			# print(points.shape) # torch.Size([16, 2500, 3])

			# create random grid
			grid = torch.FloatTensor(verts).to(opts.device)
			grid = grid.repeat(img.size(0), 1, 1)
			#grid = grid.expand(img.size(0), grid.size(1), grid.size(2))
			#grid = grid.to(opt.device)

			# forward
			points_reconstructed  = model(img, grid) # torch.Size([16, 7446, 3])

			# print(points_reconstructed.cpu().shape) # torch.Size([16, 7446, 3])
			# print(points.contiguous().cpu().shape)  # torch.Size([16, 2500, 3])

			# must match the number of points
			choice = np.random.choice(points_reconstructed.size(1), points.shape[1], replace=True)
			points_reconstructed = points_reconstructed[:,choice,:].contiguous()

			dist1, dist2 = distChamfer(points, points_reconstructed, opt.cuda)

			loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
			test_loss.update(loss_net.item())

			if i%100 == 0:
				mesh = sr.Mesh(vertices = points_reconstructed.cpu().numpy(), 
										   faces = faces,
										   tetures = None)

				

		
		print('test loss: {} '.format(test_loss.avg))