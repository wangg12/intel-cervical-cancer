from __future__ import print_function
import torch
import sys
if len(sys.argv) < 2:
  print('usage: python xx.py path/to/model.pth')
  exit()
model_path = sys.argv[1]
m = torch.load(model_path)
if 'best_model_state_dict' in m.keys():
  print('delete best_model_state_dict...')
  del m['best_model_state_dict']
else:
  print('do nothing...')
torch.save(m, model_path)

