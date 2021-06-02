import os
import shutil
import numpy as np
import re
import pandas as pd
from tqdm import tqdm

setname = "validation"
dest_dir = f"./{setname}/{setname}-set"
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

images = os.listdir(f"./{setname}/images/no_ooi")
ImageIds = [re.findall("\d+", f)[0] for f in images]
ImageIds = [int(i) for i in ImageIds]
labels = np.random.randint(2, size=len(ImageIds))

for id, label in tqdm(zip(ImageIds, labels), total=len(labels)):
    src = "ooi" if label == 1 else "no_ooi"
    src = os.path.join(f"./{setname}/images", src, f"image-z_{id}.png")
    dest = os.path.join(dest_dir, f"{id:06}.png")
    shutil.copy(src, dest)

ImageIds = [f"{i:06}" for i in ImageIds]
df = pd.DataFrame({"ImageId": ImageIds, "Label": labels})
df.to_csv(f"./{setname}/{setname}-labels.csv", index=False)
