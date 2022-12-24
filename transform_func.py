import torch 
import torchvision.transforms.functional as transF
import numpy as np
from io import BytesIO
import ctypes
from scipy.ndimage import zoom as scizoom
import skimage.color as skcolor
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import cv2
from PIL import Image as PILImage

class BrightnessTransform:
    """Adjust brightness by one of the given angles.
        @input: tensor
    """

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        x = x.permute(2,0,1) # change shape to use lib adjust_brightness
        # print('x:', x.shape)

        z = transF.adjust_brightness(x, self.brightness_factor)
 
        z = z.permute(1,2,0) # return raw shape
        # z_ = (z.numpy()*255).astype(np.uint8)
        # print('z_', z_.shape)
        # PILImage.fromarray(z_).save('convert.png')
        
        return z



# version NOT USE mean, var, scale
class AddGaussianNoise(object):
    def __call__(self, tensor):
        x = tensor
        z = torch.clamp(x + 0.02 * torch.randn_like(x), 0, 1)
        return z
    

#--------------------------------------
# Motion Blur
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def motion_blur(x): 
    """@input: PilImage
    """
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=0, sigma=8.0, angle=-60)

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)
        
    # just use for change shape
    if x.shape != (32, 32):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

class MotionBlur(object):
    def __call__(self, tensor):
        x = tensor.numpy()
        x_img = PILImage.fromarray((x*255).astype(np.uint8))
        z = motion_blur(x_img)
        z = z/255 # normalize
        z = torch.from_numpy(z).float()
        return z


#-----------------------------------------
# Snow
def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


from utils import save_img
class AddSnow(object):

    def __call__(self, tensor):
        # print('tensor input:', type(tensor), tensor.shape)
        x = tensor.numpy()
        # print('x: ', x)
        x_ = (x*255).astype(np.uint8) # just for test save img
        # print('x_', x_)
        # PILImage.fromarray(x_).save('raw.png')
        
        
        img_size = x.shape[0]
        # print('img_size: ', img_size)

        snow_layer = np.random.normal(size=(img_size, img_size), loc=0.05, scale=0.3)
        snow_layer = clipped_zoom(snow_layer[..., np.newaxis], 2)
        snow_layer[snow_layer < 0.5] = 0
        snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        output = output.getvalue()


        moving_snow = MotionImage(blob=output)
        i=30 # dont know this
        moving_snow.motion_blur(radius=10, sigma=2, angle=i*4-150)

        snow_layer = cv2.imdecode(np.fromstring(moving_snow.make_blob(), np.uint8),
                                cv2.IMREAD_UNCHANGED) / 255.
        snow_layer = snow_layer[..., np.newaxis]

        z = 0.85 * x + (1 - 0.85) * np.maximum(
            x, cv2.cvtColor(np.float32(x), cv2.COLOR_RGB2GRAY).reshape(img_size, img_size, 1) * 1.5 + 0.5)

        # print('z before uint', z)
        z = np.uint8(np.clip(z + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255)

        # save
        # print('z:', type(z), z.shape, z)
        # PILImage.fromarray(z).save('convert.png')

        z = z/255 # normalize
        z = torch.from_numpy(z).float()
        # print('z:', type(z), z.shape)
        return z 
    