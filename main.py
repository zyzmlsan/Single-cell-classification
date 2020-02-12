import numpy as np
from libtiff import TIFF
tif = TIFF.open('Control-2-2by2.tif')

stacks = list()
for t in list(tif.iter_images()):
    stacks.append(t)
stacks = np.stack(stacks)
print(stacks.shape)
print(stacks.dtype)
