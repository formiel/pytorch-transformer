import os
import argparse
import torch

def main():
    # parse parameters
    parser = argparse.ArgumentParser(description="Hang Le")
    parser.add_argument("--layerdrop", type=float, default=0.0,
                        help="LayerDrop rate")
    parser.add_argument("--amp", action='store_true',
                        help="Use apex for distributed training")
    args = parser.parse_args()
    print(args)

if __name__ == "__main__":
    main()
    