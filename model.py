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

class PointGenCon_SkipConection(nn.Module):
  def __init__(self, 
               bottleneck_size):
    
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
    
  def forward(self, x, grid):
      # x    : (#batch, #bottleneck_size)
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

class SVR_AtlasNet_SPHERE(nn.Module):
  def __init__(self,
               bottleneck_size = 1024, 
               use_pretrained_encoder = False):
    
    super(SVR_AtlasNet_SPHERE, self).__init__()
    self.use_pretrained_encoder = use_pretrained_encoder
    
    self.bottleneck_size = bottleneck_size
    self.encoder = models.resnet18(pretrained=self.use_pretrained_encoder, num_classes=self.bottleneck_size)  
    self.decoder = PointGenCon(bottleneck_size=3+self.bottleneck_size)
    
  def forward(self, x, grid):
    x = x[:,:3,:,:].contiguous()
    x = self.encoder(x)
    
    grid = grid.contiguous()
    
    y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
    y = torch.cat((grid, y), 1).contiguous()
    y = self.decoder(y).contiguous()

    return y.transpose(2,1).contiguous()

class SVR_AtlasNet_SPHERE_Plus(nn.Module):
  def __init__(self,
               bottleneck_size = 1024, 
               use_pretrained_encoder = False):
    
    super(SVR_AtlasNet_SPHERE_Plus, self).__init__()
    self.use_pretrained_encoder = use_pretrained_encoder
    
    self.bottleneck_size = bottleneck_size
    self.encoder = models.resnet18(pretrained=self.use_pretrained_encoder, num_classes=self.bottleneck_size)  
    self.decoder = PointGenCon_SkipConection(bottleneck_size=self.bottleneck_size)
    
  def forward(self, x, grid):
    x = x[:,:3,:,:].contiguous()
    x = self.encoder(x)
    
    grid = grid.contiguous()
    
    y = self.decoder(x, grid)

    return y.transpose(2,1).contiguous()

if __name__ == '__main__':
  import meshzoo

  x = torch.randn(8,3,256,256).cuda()
  vert, face = meshzoo.iso_sphere(3)

  vert = torch.FloatTensor(vert).transpose(0,1)[None,:,:].repeat(8,1,1).cuda()

  print(vert.shape)

  model = SVR_AtlasNet_SPHERE_Plus().cuda()

  print(model(x, vert).shape)