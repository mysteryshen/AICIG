"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import numpy as np
from PIL import Image

def save_image_array(img_array, fname, batch_size=100, class_num=10):
    channels = img_array.shape[1]
    resolution = img_array.shape[-1]
    img_rows = 10
    img_cols = batch_size//class_num

    img = np.full([channels, resolution * img_rows, resolution * img_cols], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[:,
            (resolution * r): (resolution * (r + 1)),
            (resolution * (c % img_cols)): (resolution * ((c % img_cols) + 1))
            ] = img_array[c+(r*img_cols)]

    img = (img * 255 + 0.5).clip(0, 255).astype(np.uint8)
    if (img.shape[0] == 1):
        img = img[0]
    else:
        img = np.rollaxis(img, 0, 3)

    Image.fromarray(img).save(fname)
