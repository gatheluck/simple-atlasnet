__all__ = [
	'PointGenCon',
	'SVR_AtlasNet_SPHERE',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PointGenCon(nn.Module):
  def __init__(self, 
               bottleneck_size):
    
    self.bottleneck_size = bottleneck_size
    super(PointGenCon, self).__init__()
    self.conv1 = torch.nn.Conv1d(self.bottleneck_size//1, self.bottleneck_size//1, 1)
    self.conv2 = torch.nn.Conv1d(self.bottleneck_size//1, self.bottleneck_size//2, 1)
    self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
    self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)
    
    self.th = nn.Tanh()
    self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size//1)
    self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
    self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
    
  def forward(self, x):
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu(self.bn3(self.conv3(x)))
      x = self.th(self.conv4(x))
      return x

class SVR_AtlasNet_SPHERE(nn.Module):
  def __init__(self, 
               num_points = 2048, 
               bottleneck_size = 1024, 
               use_pretrained_encoder = False):
    
    super(SVR_AtlasNet_SPHERE, self).__init__()
    self.num_points = num_points
    self.use_pretrained_encoder = use_pretrained_encoder
    
    self.bottleneck_size = bottleneck_size
    self.encoder = models.resnet18(pretrained=self.use_pretrained_encoder, num_classes=self.bottleneck_size)  
    self.decoder = PointGenCon(bottleneck_size=3+self.bottleneck_size)
    
  def forward(self, x, rand_grid):
    x = x[:,:3,:,:].contiguous()
    x = self.encoder(x)
    
    # rand_grid = torch.FloatTensor(x.size(0),3,self.num_points) #sample points randomly
    # rand_grid.normal_(mean=0,std=1)
    # rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points)
    
    y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
    y = torch.cat((rand_grid, y), 1).contiguous()
    y = self.decoder(y).contiguous()

    return y.transpose(2,1).contiguous()