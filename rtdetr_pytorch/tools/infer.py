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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from PIL import Image
from tqdm import tqdm


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
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
    

    class TestDataset(Dataset):
        def __init__(self, directory):
            self.root = directory
            self.files = os.listdir(directory)
            self.transform = T.Compose([
                T.Resize([640, 640]),
                T.ToImageTensor(),
                T.ConvertDtype()
            ])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            img_path = os.path.join(self.root, self.files[index])
            original_image = torchvision.datasets.folder.default_loader(img_path)
            return self.files[index], self.transform(original_image), torch.Tensor(original_image.size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    if not args.image_path:
        print("--image-path is required.")
        return

    test_dataset = TestDataset(args.image_path)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    preds = {}

    for img_ids, imgs, sizes in tqdm(dataloader):
      imgs = imgs.to(device)
      sizes = sizes.to(device)
      results = model(imgs, sizes)
      for img_id, cls, bbox, prob in zip(img_ids, *results):
        preds[img_id] = {
          'cls': cls.tolist(),
          'bbox': bbox.tolist(),
          'prob': prob.tolist()
        }
    if args.output:
      json.dump(preds, open(args.output, 'w'))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--image-path', '-p', type=str, )
    parser.add_argument('--output', '-o', type=str, )
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()

    main(args)
