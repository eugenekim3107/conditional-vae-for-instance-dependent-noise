import argparse
import os
import torch
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename-X',
    default='augmentedData/X_10_aug.pt',
    help="X data")

parser.add_argument(
    '--filename-y',
    default='augmentedData/y_aug.pt',
    help="y data")

parser.add_argument(
    '--number',
    default=np.random.randint(0,10),
    help="Number to generate")

def main(args):
    X = torch.load(args.filename_X)
    y = torch.load(args.filename_y)
    indices = (y==args.number).nonzero(as_tuple=True)[0]
    
    index = np.random.choice(indices, 1)
    image = cv2.resize(X[index].squeeze(0).cpu().detach().numpy(), (200, 200))
    
    title = f"Label: {index}"
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(parser.parse_args())
