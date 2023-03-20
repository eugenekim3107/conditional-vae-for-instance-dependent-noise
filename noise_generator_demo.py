import argparse
import os
import torch
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename-X-clean',
    default='MNIST/X.pt'
)
parser.add_argument(
    '--filename-y-clean',
    default='MNIST/y.pt'
)
parser.add_argument(
    '--filename-y-noisy',
    default='noisyData/y_10.pt',
    help='Location of label y')
parser.add_argument(
    '--number',
    default=np.random.randint(0,10),
    help='MNIST noisy number to generate')

def main(args):
    X = torch.load(args.filename_X_clean)
    y = torch.load(args.filename_y_clean)
    yn = torch.load(args.filename_y_noisy)
    number = int(args.number)
    
    indices = (y!=yn).nonzero(as_tuple=True)[0]
    
    X = X[indices]
    y = y[indices]
    yn = yn[indices]
    
    indices = (y==number).nonzero(as_tuple=True)[0]
    
    X = X[indices]
    y = y[indices]
    yn = yn[indices]
    
    n = X.shape[0]
    
    num = np.random.randint(0,n)
    
    title = f"True: {y[num]}, Noise: {yn[num]}"
    image = cv2.resize(X[num].squeeze(0).numpy(), (200, 200))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(parser.parse_args())