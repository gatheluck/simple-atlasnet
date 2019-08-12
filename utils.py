__all__ = [
	'weights_init',
  'my_get_n_random_lines',
  'AverageValueMeter',
]

import os
import random

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