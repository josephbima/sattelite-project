from Synthesizer import Synthesizer
import pandas as pd
from tqdm import trange
import random

### This class is so incomplete
class DataGenerator:
    def __init__(self, n, camera_function: pd.DataFrame):
        self.n = n
        self.synth = Synthesizer(camera_function)

    def generate_data(self, im_size,template='test', start=0):
        for i in trange(start,self.n+start):
            try:
                self.synth.generate_random_files(im_size[0]*im_size[1], selection_pickle='file_selection_dataframe/success.pkl')
                self.synth.generate_img_from_file(dim=im_size, filename=f'generated_images/{template}_{i}.tiff')
            except ValueError:
                continue
    
    def generate_voronoi(self, im_size, template='voronoi', start=0, min_num_cells=10 ,max_num_cells=20, min_num_materials=5, max_num_materials=10):

        # Add selection data pickle
        self.synth.generate_random_files(n=1, selection_pickle = 'file_selection_dataframe/success.pkl')
        
        for i in trange(start,self.n+start):
            try:
                rand_num_cells = random.randrange(min_num_cells, max_num_cells)
                rand_num_materials = random.randrange(min_num_materials, max_num_materials)
                fn = f'generated_images/{template}_{i}.tiff'
                self.synth.generate_voronoi(im_size[0], im_size[1], rand_num_cells, rand_num_materials,filename=fn, sample_random=True)
            except ValueError:
                continue



camera_df = pd.read_pickle('normalized_df.pkl')
dg = DataGenerator(500,camera_df)
dg.generate_voronoi((1000,1000), min_num_cells=10, max_num_cells=25,min_num_materials=10, max_num_materials=20)
