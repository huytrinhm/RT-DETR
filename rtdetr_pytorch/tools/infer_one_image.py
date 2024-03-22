"""by lyuwenyu
"""

import os 
import json
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm

def load_model(config, resume_path):
    cfg = YAMLConfig(config, resume=resume_path)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    return model


def detect_one_image(model, image_path):
    transform = T.Compose([
        T.Resize([640, 640]),
        T.ToImageTensor(),
        T.ConvertDtype()
    ])

    img = Image.open(image_path)
    size = torch.Tensor(img.size)
    img = transform(img)
    results = model(img.unsqueeze(0), size.unsqueeze(0))

    preds = {
      'cls': results[0][0].tolist(),
      'bbox': results[1][0].tolist(),
      'prob': results[2][0].tolist()
    }

    return preds


def main(args, ):
    """main
    """
    model = load_model(args.config, args.resume)

    if not args.image_path:
        print("--image-path is required.")
        return

    preds = detect_one_image(model, args.image_path)

    if args.output:
      json.dump(preds, open(args.output, 'w'))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--image-path', '-p', type=str, )
    parser.add_argument('--output', '-o', type=str, )

    args = parser.parse_args()

    main(args)
