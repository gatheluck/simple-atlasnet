from dataset import *
from options import *
from model import *
from utils import *

import os
import sys

import numpy as np

import torch
import torch.optim as optim

import visdom

if __name__ == '__main__':

	opt = TrainOptions().parse()
	vis = visdom.Visdom(port = 8888)

	dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.class_choice, train=opt.use_train)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))
	len_dataset = len(dataset)
	print('training set: ', len_dataset)

	model = SVR_AtlasNet_SPHERE()
	model.apply(weights_init)
	model = model.to(opt.device)


	if opt.fix_decoder:
		optimizer = optim.Adam(model.encoder.parameters(), lr = opt.lr)
	else:
		optimizer = optim.Adam(model.parameters(), lr = opt.lr)

	# logger
	train_loss = AverageValueMeter()
	train_curve = []

	for epoch in range(opt.num_epochs):
		train_loss.reset()
		model.train()

		if epoch==100:
			if opt.fix_decoder:
				optimizer = optim.Adam(model.encoder.parameters(), lr = opt.lr/10.0)
			else:
				optimizer = optim.Adam(model.parameters(), lr = opt.lr/10.0)

		for i, data in enumerate(dataloader, 0):
			optimizer.zero_grad()   
			img, points, label, _ , _= data
			img, points = img.to(opt.device), points.to(opt.device)

			# create random grid
			rand_grid = torch.FloatTensor(img.size(0),3,opt.num_points) #sample points randomly
			rand_grid.normal_(mean=0,std=1)
			rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(img.size(0),3,opt.num_points)
			rand_grid = rand_grid.to(opt.device)

			# forward
			points_reconstructed  = model(img, rand_grid)
			# print(points_reconstructed.cpu().shape) # torch.Size([32, 2500, 3])
			# print(points.transpose(2,1).contiguous().cpu().shape) # torch.Size([32, 2500, 3])

			dist1, dist2 = distChamfer(points, points_reconstructed, opt.cuda)

			loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
			loss_net.backward()
			optimizer.step()

			if i % opt.vis_freq == 0:
				vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
				vis.scatter(X = points[0].data.cpu(),
										win = 'TRAIN_INPUT',
										opts = dict(title="TRAIN_INPUT", markersize=2)
				)
				vis.scatter(X = points_reconstructed[0].data.cpu(),
										win = 'TRAIN_INPUT_RECONSTRUCTED',
										opts = dict(title="TRAIN_INPUT_RECONSTRUCTED", markersize=2)
				)

			train_loss.update(loss_net.item())
			print('[{}: {}/{}] train loss: {} '.format(epoch, i, int(len_dataset/opt.batch_size), loss_net.item()))

		if epoch % opt.save_freq == 0:
			save_model(model, os.path.join(opt.log_dir, 'weight_{:03d}.pth'.format(epoch+1)))

		save_model(model, os.path.join(opt.log_dir, 'weight_tmp.pth'))

		train_curve.append(train_loss.avg)
		vis.line(X=np.arange(len(train_curve)),
						 Y=np.array(train_curve),
						 win='loss',
						 opts=dict(title="loss", legend=["train_curve"], markersize=2))

	save_model(model, os.path.join(opt.log_dir, 'weight_final.pth'))