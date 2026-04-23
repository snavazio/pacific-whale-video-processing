from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch import argmax
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from model.model_base import model_base

# implementation of model on pytorch.
class model_pytorch(model_base):
  def __init__(self, model_name, num_features=2):
    model_base.__init__(self) # init base class

    # Pick the best available device. CPU is a valid fallback on machines
    # without CUDA — previously a hardcoded call to set_default_tensor_type
    # forced cuda tensors and crashed CPU-only hosts.
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = self.get_resnet18(num_features)
    model_ft = model_ft.to(self.device)
    # map_location ensures weights saved on a GPU load cleanly on a CPU host.
    # weights_only=True is the safe mode for loading state dicts (avoids arbitrary
    # pickle execution) and will become the default in a future PyTorch release.
    model_ft.load_state_dict(torch.load(model_name, map_location=self.device, weights_only=True))
    model_ft.eval() #this sets the model to "evaluate" mode.

    self.model = model_ft
    self.num_features = num_features

    # set transformation - this will largely depend on the model and the transforms used during training.
    self.transforms = transforms.Compose([     
      transforms.Resize(256),
      transforms.CenterCrop(224),        
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print("Finished loading model from file {0}".format(model_name))

  def get_number_of_features(self):
    return self.num_features

  def predict(self, image_path):
    image_pil = Image.open(image_path)
    input_image = self.transforms(image_pil).view(1,3,224,224).to(self.device)
    model_pred = self.model(input_image)
    _,preds = torch.max(model_pred,1)
    return preds[0]

  def get_resnet18(self, num_features):
    # weights=None constructs a randomly-initialized ResNet18. We immediately
    # overwrite its state in __init__ via load_state_dict, so downloading the
    # ImageNet-pretrained weights (what pretrained=True / weights=DEFAULT did)
    # would be ~46MB of wasted I/O on every first run. The deprecated
    # pretrained=True kwarg was also removed in newer torchvision versions.
    temp_model = models.resnet18(weights=None)
    num_ftrs = temp_model.fc.in_features
    temp_model.fc = nn.Linear(num_ftrs, num_features)
    return temp_model
