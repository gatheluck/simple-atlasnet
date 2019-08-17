__all__ = [
	'TrainOptions',
	'TestOptions'
]

import os
import sys
import random
import argparse

import torch
from torchvision import models

classes = ['plane',
					'bench',
					'cabinet',
					'car',
					'chair',
					'monitor',
					'lamp',
					'speaker',
					'firearm',
					'couch',
					'table',
					'cellphone',
					'watercraft'
]

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		# model
		parser.add_argument('--pretrained_enc', action='store_true', default=False, help='use pre-trained encoder')
		parser.add_argument('--skipconnected_dec', action='store_true', default=True, help='use skip connection')
		# dataset
		parser.add_argument('-j', '--num_workers', type=int, default=4, help='number of workers for data loading')
		parser.add_argument('-N', '--batch_size', type=int, default=16, help='batch size')
		parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
		parser.add_argument('--use_train', action='store_true', default=True, help='if use data for training or test')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')
		# log
		parser.add_argument('-l', '--log_dir', type=str, required=True, help='log directory')
		parser.add_argument('--output_freq', type=int, default=-1, help='output frequences')
		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')
	
	def parse(self):
		opt = self.gather_options()

		if opt.num_classes <=0 or opt.num_classes >= len(classes):
			opt.class_choice = classes
		else:
			opt.class_choice = classes[:opt.num_classes]

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.cuda = False
			opt.device = 'cpu'

		self.opt = opt
		return self.opt
			

class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		
		# model
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		parser.add_argument('--fix_decoder', action='store_true', default=False, help='enable GPU')
		parser.add_argument('--save_freq', type=int, default=30, help='save frequence')
		parser.add_argument('--num_points', type=int, default=2500, help='number of sampling points')
		# visualization
		parser.add_argument('--vis_freq', type=int, default=50, help='visualization frequences of results')
		# hyperparameter
		parser.add_argument('--num_epochs', type=int, default=90, help='number of epochs')
		parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
		
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt

class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		
		# model
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt