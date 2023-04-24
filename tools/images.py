
import io
from PIL import Image as PILImage
import kornia
import torch
import numpy as np

from .constants import SOURCES

def prep_image(name, size=512, crop_square=True, crop_offset=0.5,  mask=None):
    path = next(SOURCES.glob(f"{name}.*"))
    im = PILImage.open(path)

    w, h = im.width, im.height
    scale = size / min(w, h)
    dw, dh = max(size, round(w * scale)), max(size, round(h * scale))
    im = im.resize((dw, dh))

    if crop_square:
        w, h = im.width, im.height
        l, t = round((w-size) * crop_offset), round((h-size) * crop_offset)
        im = im.crop((l, t, l+size, t+size))

    if mask is not None:
        maskim, _ = prep_image(f"{name}_mask_{mask}", size, crop_square, crop_offset)
        im.putalpha(maskim if maskim.mode == "L" else maskim.getchannel("R"))

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)

    return im, buf

def ssim(path1, path2):
    image1 = np.array(PILImage.open(path1)).astype(np.float32) / 255.0
    image2 = np.array(PILImage.open(path2)).astype(np.float32) / 255.0

    tensor1 = kornia.utils.image_to_tensor(image1, keepdim=False)
    tensor2 = kornia.utils.image_to_tensor(image2, keepdim=False)

    return kornia.losses.ssim_loss(tensor1, tensor2, 21)
