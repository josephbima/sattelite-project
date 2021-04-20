#!/Users/bimajenie/opt/anaconda3/bin/env python3
import pandas as pd
import cv2
from mlib import *
from Synthesizer import Synthesizer
import rasterio as rio
import numpy as np

camera_df = pd.read_pickle('camera_df.pkl')

synth = Synthesizer(camera_df)
#
# synth.generate_random_files(25, selection_pickle='success.pkl')
# synth.generate_img_from_file(dim=(5,5), filename='test1.tiff')
#
# synth.generate_random_files(25)
# synth.generate_img_from_file(dim=(5,5), filename='test2.tiff')
# #
# synth.generate_random_files(100, selection_pickle='success.pkl')
# synth.generate_img_from_file(dim=(10,10), filename=r'generated_images/test08.tiff')

## RGB Testing
# rgb_df = pd.read_pickle('rgb_df.pkl')
# rgb = Synthesizer(rgb_df)
#
# rgb.generate_random_files(25, selection_pickle='success.pkl')
# rgb.generate_img_from_file(dim=(5,5), filename='test_rgb.tiff')
#
# im = rgb.img_arr[:,:, :3]
# cv2.imshow('im', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# im = rasterio.open(r'generated_images/test50.tiff').read()
# print(im.shape)
# cv2.imshow('im',im[3,:,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(1000,1200):
#     synth.generate_random_files(100, selection_pickle='success.pkl')
#     synth.generate_img_from_file(dim=(10,10), filename=f'generated_images/test{i}.tiff')

# synth.generate_random_files(100, selection_pickle='success.pkl')
# synth.generate_img_from_file(dim=(10,10), filename=f'generated_images/test-1.tiff')

smp = rio.open(r'generated_images/test37.tiff').read()
bands=[]
for i in range(12):
    bands.append(mcv.imshow(smp, multiply=1, channel_first=True, bands=i, ret=True, show=False))
mcv.batch_show(bands)
mcv.imshow(smp, multiply=1, channel_first=True)


