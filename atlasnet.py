__all__ = [
	'PointGenCon',
  'PointGenCon_SkipConection',
	'SVR_AtlasNet_SPHERE',
  'SVR_AtlasNet_SPHERE_Plus',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PointGenDecoder(nn.Module):
  def __init__(self, 
               latent_size):
    
    self.bottleneck_size = bottleneck_size
    super(PointGenCon_SkipConection, self).__init__()
    self.conv1 = torch.nn.Conv1d((self.bottleneck_size//1)+3, self.bottleneck_size//1, kernel_size=1)
    self.conv2 = torch.nn.Conv1d((self.bottleneck_size//1)+3, self.bottleneck_size//2, kernel_size=1)
    self.conv3 = torch.nn.Conv1d((self.bottleneck_size//2)+3, self.bottleneck_size//4, kernel_size=1)
    self.conv4 = torch.nn.Conv1d((self.bottleneck_size//4)+3, 3, kernel_size=1)
    
    self.th = nn.Tanh()
    self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size//1)
    self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
    self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
    
  def forward(self, z, grid):
      # z    : (#batch, #bottleneck_size)
      # grid : (#batch, 3, #points)

      x = x[:,:,None].repeat(1,1,grid.size(2)) # x : (#batch, #bottleneck_size, #points)

      x = torch.cat((x, grid), 1) 
      x = F.relu(self.bn1(self.conv1(x)))

      x = torch.cat((x, grid), 1)
      x = F.relu(self.bn2(self.conv2(x)))

      x = torch.cat((x, grid), 1)
      x = F.relu(self.bn3(self.conv3(x)))

      x = torch.cat((x, grid), 1)
      x = self.th(self.conv4(x))
      return x

class PointGenDecoderSkipconnected(nn.Module):
	def __init__(self, latent_size):
    
    self.bottleneck_size = bottleneck_size
    super(PointGenCon_SkipConection, self).__init__()
    self.conv1 = torch.nn.Conv1d((self.bottleneck_size//1)+3, self.bottleneck_size//1, kernel_size=1)
    self.conv2 = torch.nn.Conv1d((self.bottleneck_size//1)+3, self.bottleneck_size//2, kernel_size=1)
    self.conv3 = torch.nn.Conv1d((self.bottleneck_size//2)+3, self.bottleneck_size//4, kernel_size=1)
    self.conv4 = torch.nn.Conv1d((self.bottleneck_size//4)+3, 3, kernel_size=1)
    
    self.th = nn.Tanh()
    self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size//1)
    self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
    self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
    
  def forward(self, z, grid):
      # z    : (#batch, #bottleneck_size)
      # grid : (#batch, 3, #points)

      x = x[:,:,None].repeat(1,1,grid.size(2)) # x : (#batch, #bottleneck_size, #points)

      x = torch.cat((x, grid), 1) 
      x = F.relu(self.bn1(self.conv1(x)))

      x = torch.cat((x, grid), 1)
      x = F.relu(self.bn2(self.conv2(x)))

      x = torch.cat((x, grid), 1)
      x = F.relu(self.bn3(self.conv3(x)))

      x = torch.cat((x, grid), 1)
      x = self.th(self.conv4(x))
      return x

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
    #self.decoder = PointGenCon_SkipConection(bottleneck_size=self.bottleneck_size)
    
  def forward(self, im, verts):
		assert len(im.size()) == 4,    'im:    (#batch, 3(rgb), #hight, #width)'
		assert len(verts.size()) == 3, 'verts: (#batch, 3(xyz), #vertex)'

    im = im[:,:3,:,:] # remove alpha
    z = self.encoder(im) # z: (#batch, latent_size)
    
    
    
    y = self.decoder(z, grid)

    return y.transpose(2,1).contiguous()