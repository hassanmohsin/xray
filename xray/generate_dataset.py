from argparse import ArgumentParser
from time import time

import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm


def argument_parser():
    parser = ArgumentParser(description='Generate xray image dataset')
    parser.add_argument('--images', type=str, required=True, action='store', help="Image directory")
    parser.add_argument('--set', type=str, required=True, action='store', help="Name of the set e.g., 'train'")
    parser.add_argument('--multiview', action='store_true', default=False, help="Multi-view dataset")
    parser.add_argument('--split', type=float, required=True, action='store',
                        help="Ratio of positive to negative samples")
    parser.add_argument('--output_dir', type=str, action='store', default='new-dataset',
                        help="Output directory")
    args = parser.parse_args()
    tic = time()

    images = os.listdir(os.path.join(args.images, f"{args.set}", "no_ooi", "zview"))
    image_ids = [os.path.splitext(i)[0] for i in images]
    labels = np.random.uniform(size=len(image_ids)) > args.split

    if not args.multiview:
        dest_dir = f"./{args.output_dir}/{args.set}-set"
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        for id, label in tqdm(zip(image_ids, labels), total=len(labels)):
            src = "ooi/zview" if label else "no_ooi/zview"
            src = os.path.join(args.images, args.set, src, f"{id}.png")
            dest = os.path.join(dest_dir, f"{id}.png")
            shutil.copy(src, dest)
    else:
        parent_dir = os.path.join(args.output_dir, f"{args.set}")
        views = ['xview', 'yview', 'zview']
        views_dirs = [os.path.join(parent_dir, v) for v in views]
        # Create directories
        for vdir in views_dirs:
            if not os.path.isdir(vdir):
                os.makedirs(vdir)
            else:
                raise IsADirectoryError("Old directory exists.")

        for id, label in tqdm(zip(image_ids, labels), total=len(labels)):
            for view in views:
                dest_dir = os.path.join(parent_dir, view)
                src = f"ooi/{view}" if label else f"no_ooi/{view}"
                src = os.path.join(args.images, f"{args.set}", src, f"{id}.png")
                dest = os.path.join(dest_dir, f"{id}.png")
                shutil.copy(src, dest)

    labels = [int(i) for i in labels]
    df = pd.DataFrame({"image_id": image_ids, "label": labels})
    df.to_csv(os.path.join(args.output_dir, f"{args.set}-labels.csv"), index=False)
    print(f"Execution time: {time() - tic} seconds.")


if __name__ == '__main__':
    argument_parser()
