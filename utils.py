__all__ = [
  'render_as_gif'
	'weights_init',
  'my_get_n_random_lines',
  'distChamfer',
  'save_model',
  'AverageValueMeter',
]

import os
import sys
import random

import torch
import torch.nn as nn

import imageio
import tqdm
import soft_renderer as sr

def render_as_gif(verts, faces,
                  output_path,
                  camera_distance = 3.0,
                  elevation = 30.0,
                  rotation_delta = 4,
                  verbose = False):
  assert len(verts.size()) == 3
  assert len(faces.size()) == 3
  output_path = os.path.splitext(output_path)[0] + '.git'  # replace extention by .git  
  os.makedirs(os.path.dirname(output_path), exist_ok=True) # make output dir

  mesh = sr.Mesh(verts, faces)
  renderer = sr.SoftRenderer(camera_mode='look_at')
  writer = imageio.get_writer(output_path, mode='I')

  loop = list(range(0, 360, rotation_delta))
  loop = tqdm.tqdm(loop) if verbose else loop

  for idx, azimuth in enumerate(loop):
    mesh.reset_()
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    imgs = renderer.render_mesh(mesh)
    img  = imgs.detach().cpu()[0,:,:,:].numpy().transpose((1, 2, 0))
    writer.append_data((255*img).astype(np.uint8))
  writer.close()
    

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

lenght_line = 60
def my_get_n_random_lines(path, n=5):
  MY_CHUNK_SIZE = lenght_line * (n+2)
  lenght = os.stat(path).st_size
  
  with open(path, 'r') as file:
    file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
    chunk = file.read(MY_CHUNK_SIZE)
    
    # lines = chunk.split(os.linesep)
    lines = chunk.split('\n')
    
    return lines[1:n+1]

def distChamfer(a,b,cuda=True):
  x,y = a,b
  bs, num_points, points_dim = x.size()
  xx = torch.bmm(x, x.transpose(2,1))
  yy = torch.bmm(y, y.transpose(2,1))
  zz = torch.bmm(x, y.transpose(2,1))
  if cuda is True:
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
  else:
    diag_ind = torch.arange(0, num_points).type(torch.LongTensor)
  rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
  ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
  P = (rx.transpose(2,1) + ry - 2*zz)
  return P.min(1)[0], P.min(2)[0]

def save_model(model, path):
  torch.save(
    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    path
  )

class AverageValueMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0.0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

if __name__ == "__main__":
  import meshzoo

  verts, faces = meshzoo.iso_sphere(3)
  output_path = 'logs/test_gif_01.gif'
  render_as_gif(verts, faces, output_path, verbose=True)

  output_path = 'logs/test_gif_02'
  render_as_gif(verts, faces, output_path, verbose=False)