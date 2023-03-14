import argparse
import os
import torch
import numpy
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename-X',
    default='noisyData/X_10.pt',
    help='Location of features X')
parser.add_argument(
    '--filename-y',
    default='noisyData/y_10.pt',
    help='Location of label y')
parser.add_argument(
    '--number',
    default=np.random.randint(0,10,1),
    help='MNIST noisy number to generate'

def main(args):
    X = torch.load(args.filename_X)
    y = torch.load(args.filename_y)
    
    print(X)
    print(y)
    
    
    
    
if __name__ == "__main__":
    main(parser.parse_args())