import json
import os
import re
import shutil
from argparse import ArgumentParser
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def argument_parser():
    parser = ArgumentParser(description='Generate xray image dataset.')
    parser.add_argument('--input', type=str, required=True, action='store',
                        help="JSON input")
    parser.add_argument('--set', type=str, required=True, action='store', help="Name of the set e.g., 'train'")
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError("Input {args.input} not found.")

    with open(args.input) as f:
        image_args = json.load(f)

    tic = time()

    dest_dir = f"./{args.set}/{args.set}-set"
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    images = os.listdir(f"./{args.set}/images/no_ooi")
    ImageIds = [re.findall("\d+", f)[0] for f in images]
    ImageIds = [int(i) for i in ImageIds]
    labels = np.random.randint(2, size=len(ImageIds))

    for id, label in tqdm(zip(ImageIds, labels), total=len(labels)):
        src = "ooi" if label == 1 else "no_ooi"
        src = os.path.join(f"./{args.set}/images", src, f"image-z_{id}.png")
        dest = os.path.join(dest_dir, f"{id:06}.png")
        shutil.copy(src, dest)

    ImageIds = [f"{i:06}" for i in ImageIds]
    df = pd.DataFrame({"ImageId": ImageIds, "Label": labels})
    df.to_csv(f"./{args.set}/{args.set}-labels.csv", index=False)

    print(f"Execution time: {time() - tic} seconds.")


if __name__ == '__main__':
    argument_parser()
