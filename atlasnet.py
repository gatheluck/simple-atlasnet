__all__ = [
	'PointGenCon',
  'PointGenSkipconnectedDecoder',
	'AtlasNetSingle',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PointGenDecoder(nn.Module):
	def __init__(self, latent_size):
    
    self.latent_size = latent_size
    super(PointGenDecoder, self).__init__()
    self.conv1 = torch.nn.Conv1d(self.latent_size//1, self.latent_size//1, kernel_size=1)
    self.conv2 = torch.nn.Conv1d(self.latent_size//1, self.latent_size//2, kernel_size=1)
    self.conv3 = torch.nn.Conv1d(self.latent_size//2, self.latent_size//4, kernel_size=1)
    self.conv4 = torch.nn.Conv1d(self.latent_size//4, 3, kernel_size=1)
    
    self.th = nn.Tanh()
    self.bn1 = torch.nn.BatchNorm1d(self.latent_size//1)
    self.bn2 = torch.nn.BatchNorm1d(self.latent_size//2)
    self.bn3 = torch.nn.BatchNorm1d(self.latent_size//4)
    
  def forward(self, z, verts):
      # z     : (#batch, #latent_size)
      # verts : (#batch, 3, #vertex)

			num_verts = verts.size(2)
      z = z[:,:,None].repeat(1,1,num_verts) # z: (#batch, #latent_size, #vertex)

      y = torch.cat((z, verts), 1) # z: (#batch, #latent_size+3, #vertex)
      y = F.relu(self.bn1(self.conv1(y)))
      y = F.relu(self.bn2(self.conv2(y)))
      y = F.relu(self.bn3(self.conv3(y)))
      y = self.th(self.conv4(y))

			# y: (#batch, 3, #vertex)
      return y


class PointGenSkipconnectedDecoder(nn.Module):
	def __init__(self, latent_size):
    
    self.latent_size = latent_size
    super(PointGenSkipconnectedDecoder, self).__init__()
    self.conv1 = torch.nn.Conv1d((self.latent_size//1)+3, self.latent_size//1, kernel_size=1)
    self.conv2 = torch.nn.Conv1d((self.latent_size//1)+3, self.latent_size//2, kernel_size=1)
    self.conv3 = torch.nn.Conv1d((self.latent_size//2)+3, self.latent_size//4, kernel_size=1)
    self.conv4 = torch.nn.Conv1d((self.latent_size//4)+3, 3, kernel_size=1)
    
    self.th = nn.Tanh()
    self.bn1 = torch.nn.BatchNorm1d(self.latent_size//1)
    self.bn2 = torch.nn.BatchNorm1d(self.latent_size//2)
    self.bn3 = torch.nn.BatchNorm1d(self.latent_size//4)
    
  def forward(self, z, verts):
      # z     : (#batch, #latent_size)
      # verts : (#batch, 3, #vertex)

			num_verts = verts.size(2)
      z = z[:,:,None].repeat(1,1,num_verts) # z: (#batch, #latent_size, #vertex)

      y = torch.cat((z, verts), 1) # y: (#batch, #latent_size+3, #vertex)
      y = F.relu(self.bn1(self.conv1(y)))

      y = torch.cat((y, verts), 1)
      y = F.relu(self.bn2(self.conv2(y)))

      y = torch.cat((y, verts), 1)
      y = F.relu(self.bn3(self.conv3(y)))

      y = torch.cat((y, verts), 1)
      y = self.th(self.conv4(y))

			# y: (#batch, 3, #vertex)
      return y


class AtlasNetSingle(nn.Module):
  def __init__(self,
               latent_size = 1024, 
               use_pretrained_encoder = False,
							 use_skipconnected_decoder = True):
    
    super(AtlasNetSingle, self).__init__()
    self.latent_size = latent_size
    self.use_pretrained_encoder = use_pretrained_encoder
		self.use_skipconnected_decoder = use_skipconnected_decoder

    self.encoder = models.resnet18(pretrained=self.use_pretrained_encoder, num_classes=self.latent_size)
		self.decoder = PointGenSkipconnectedDecoder(latent_size=self.latent_size) if self.use_skipconnected_decoder else PointGenDecoder(latent_size=self.latent_size)
    
  def forward(self, im, verts):
		assert len(im.size()) == 4,    'im:    (#batch, 3(rgb), #hight, #width)'
		assert len(verts.size()) == 3, 'verts: (#batch, 3(xyz), #vertex)'

    im = im[:,:3,:,:] # remove alpha
    z = self.encoder(im) # z: (#batch, latent_size)
    
    y = self.decoder(z, verts) # y: (#batch, 3, #vertex)

    return y