from PIL import ImageFilter
from PIL import Image
import random
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, t2, t3):
        self.base_transform = base_transform
        self.t2 = t2
        self.t3 = t3

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        # YOCO in PIL image level
        if np.random.random() < 0.5:
            new_q= Image.new('RGB', (224, 224))
            new_q.paste(self.t2(q.crop((0,0,224,112))), (0, 0))
            new_q.paste(self.t2(q.crop((0,112,224,224))), (0, 112))
            new_q = self.t3(new_q)
            new_k= Image.new('RGB', (224, 224))
            new_k.paste(self.t2(k.crop((0,0,224,112))), (0, 0))
            new_k.paste(self.t2(k.crop((0,112,224,224))), (0, 112))
            new_k = self.t3(new_k)
        else:
            new_q= Image.new('RGB', (224, 224))
            new_q.paste(self.t2(q.crop((0,0,112,224))), (0, 0))
            new_q.paste(self.t2(q.crop((112,0,224,224))), (112, 0))
            new_q = self.t3(new_q)
            new_k= Image.new('RGB', (224, 224))
            new_k.paste(self.t2(k.crop((0,0,112,224))), (0, 0))
            new_k.paste(self.t2(k.crop((112,0,224,224))), (112, 0))
            new_k = self.t3(new_k)

        return [new_q, new_k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
