from dataset import *
from options import *
from model import *
from utils import *

import torch
import torch.optim as optim

if __name__ == '__main__':

	opt = TrainOptions().parse()

	dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.class_choice, train=opt.use_train)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers))

	print('training set: ', len(dataset.datapath))

	model = SVR_AtlasNet_SPHERE(num_points = opt.num_points)
	model.apply(weights_init)
	model = model.to(opt.device)


	if opt.fix_decoder:
		optimizer = optim.Adam(model.encoder.parameters(), lr = opt.lr)
	else:
		optimizer = optim.Adam(model.parameters(), lr = opt.lr)


	for epoch in range(opt.num_epochs):
		model.train()
		if epoch==100:
			if opt.fix_decoder:
				optimizer = optim.Adam(model.encoder.parameters(), lr = opt.lr/10.0)
			else:
				optimizer = optim.Adam(model.parameters(), lr = opt.lr/10.0)

		for i, data in enumerate(dataloader, 0):
			optimizer.zero_grad()   
			img, points, label, _ , _= data
			points = points.transpose(2,1).contiguous()
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

			dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), points_reconstructed, opt.cuda)

			


