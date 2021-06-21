import os
import shutil
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def argument_parser():
    parser = ArgumentParser(description='Generate xray image dataset.')
    parser.add_argument('--images', type=str, required=True, action='store', help="Image directory")
    parser.add_argument('--set', type=str, required=True, action='store', help="Name of the set e.g., 'train'")
    parser.add_argument('--split', type=float, required=True, action='store',
                        help="Ratio of positive to negative samples")
    args = parser.parse_args()
    tic = time()

    dest_dir = f"./{args.set}/{args.set}-set"
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    images = os.listdir(f"./{args.images}/no_ooi/zview")
    ImageIds = [os.path.splitext(i)[0] for i in images]
    labels = np.random.uniform(size=len(ImageIds)) > args.split

    for id, label in tqdm(zip(ImageIds, labels), total=len(labels)):
        src = "ooi/zview" if label else "no_ooi/zview"
        src = os.path.join(args.images, src, f"{id}.png")
        dest = os.path.join(dest_dir, f"{id}.png")
        shutil.copy(src, dest)
    
    labels = [int(i) for i in labels]
    df = pd.DataFrame({"ImageId": ImageIds, "Label": labels})
    df.to_csv(f"./{args.set}/{args.set}-labels.csv", index=False)

    print(f"Execution time: {time() - tic} seconds.")


if __name__ == '__main__':
    argument_parser()

