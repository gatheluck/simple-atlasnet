__all__ = [
  'save_as_obj',
  'save_as_gif',
	'weights_init',
  'my_get_n_random_lines',
  'distChamfer',
  'save_model',
  'AverageValueMeter',
]

import os
import sys
import random

import numpy as np

import torch
import torch.nn as nn

import imageio
import tqdm
import soft_renderer as sr

def save_as_obj(verts, faces, output_path, verbose = False):
  assert len(verts.shape) == 3
  assert len(faces.shape) == 3
  if torch.cuda.is_available() is not True: 
    return None # soft renderer is only supported under cuda
  else:
    if verbose: print("saving as obj...")

  # prepare for output
  output_path = os.path.splitext(output_path)[0] + '.obj'  # replace extention by .obj
  os.makedirs(os.path.dirname(output_path), exist_ok=True) # make output dir
  if verbose: print("output_path: {}".format(output_path))

  # make mesh
  mesh = sr.Mesh(verts[0,:,:], faces[0,:,:])
  mesh.save_obj(output_path)


def save_as_gif(verts, faces,
                output_path,
                input_img = None,
                camera_distance = 3.0,
                elevation = 30.0,
                rotation_delta = 4,
                verbose = False):
  assert len(verts.shape) == 3
  assert len(faces.shape) == 3
  assert input_img is None or len(input_img.shape) == 4
  if torch.cuda.is_available() is not True: 
    return None # soft renderer is only supported under cuda
  else:
    if verbose: print("saving as gif...")

  # downsample, rescale and transpose input_img
  if input_img is not None:
    input_img = nn.Upsample((256,256), mode='bilinear')(input_img)
    input_img = 255*input_img if torch.max(input_img) <= 1.0 else input_img 
    input_img = input_img[0,:,:,:].cpu().numpy().transpose((1,2,0))
  else:
    input_img = None

  # prepare for output
  output_path = os.path.splitext(output_path)[0] + '.gif'  # replace extention by .gif
  os.makedirs(os.path.dirname(output_path), exist_ok=True) # make output dir
  writer = imageio.get_writer(output_path, mode='I')
  if verbose: print("output_path: {}".format(output_path))

  # make mesh and renderer
  mesh = sr.Mesh(verts[0,:,:], faces[0,:,:])
  renderer = sr.SoftRenderer(camera_mode='look_at')

  loop = list(range(0, 360, rotation_delta))
  loop = tqdm.tqdm(loop) if verbose else loop

  for idx, azimuth in enumerate(loop):
    mesh.reset_()
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)

    # render
    imgs = renderer.render_mesh(mesh)

    img  = imgs.detach().cpu()[0,:,:,:].numpy().transpose((1, 2, 0))*255
    img  = np.concatenate((input_img, img[:,:,:3]), axis=1) if input_img is not None else img
    
    writer.append_data((img).astype(np.uint8))
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
  import numpy as np
  import meshzoo

  verts, faces = meshzoo.iso_sphere(3)
  verts = verts[np.newaxis,...]
  faces = faces[np.newaxis,...]

  output_path = 'logs/test_gif_01.gif'
  save_as_gif(verts, faces, output_path, input_img=None, verbose=True)

  output_path = 'logs/test_gif_02'
  save_as_gif(verts, faces, output_path, input_img=None, verbose=False)
  save_as_obj(verts, faces, output_path, verbose=True)

  input_img = torch.randn(16,3,300,300).to("cuda")
  output_path = 'logs/test_gif_03'
  save_as_gif(verts, faces, output_path, input_img=input_img, verbose=False)