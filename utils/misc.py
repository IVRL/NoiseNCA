import io
import base64
import requests
import numpy as np
from PIL import Image
# import matplotlib.pyplot as pl


def imread(url, max_size=None, mode=None):
    if url.startswith(('http:', 'https:')):
        # wikimedia requires a user agent
        headers = {
            "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    if mode is not None:
        img = img.convert(mode)
    img = np.float32(img) / 255.0
    return img


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt='jpeg'):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode('ascii')
    return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string


def imshow(a, fmt='jpeg', id=None):
    return display(Image(data=imencode(a, fmt)), display_id=id)


# def grab_plot(close=True):
#     """Return the current Matplotlib figure as an image"""
#     fig = pl.gcf()
#     fig.canvas.draw()
#     img = np.array(fig.canvas.renderer._renderer)
#     a = np.float32(img[..., 3:] / 255.0)
#     img = np.uint8(255 * (1.0 - a) + img[..., :3] * a)  # alpha
#     if close:
#         pl.close()
#     return img


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img
