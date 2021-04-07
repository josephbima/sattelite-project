#!/Users/bimajenie/opt/anaconda3/bin/env python3
import pandas as pd
import cv2
from mlib import *
import matplotlib.pyplot as plt

from Synthesizer import Synthesizer

camera_df = pd.read_pickle('camera_df.pkl')

synth = Synthesizer(camera_df)

synth.generate_random_files(25, selection_pickle='success.pkl')
synth.generate_img_from_file(dim=(5,5), filename='test1.tiff')

synth.generate_random_files(25)
synth.generate_img_from_file(dim=(5,5), filename='test2.tiff')
#
synth.generate_random_files(100, selection_pickle='success.pkl')
synth.generate_img_from_file(dim=(10,10), filename=r'generated_images/test08.tiff')

## RGB Testing
rgb_df = pd.read_pickle('rgb_df.pkl')
rgb = Synthesizer(rgb_df)

rgb.generate_random_files(25, selection_pickle='success.pkl')
rgb.generate_img_from_file(dim=(5,5), filename='test_rgb.tiff')

im = rgb.img_arr[:,:, :3]
cv2.imshow('im', im)
cv2.waitKey(0)
cv2.destroyAllWindows()