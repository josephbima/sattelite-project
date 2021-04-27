from Synthesizer import Synthesizer
import pandas as pd
from tqdm import trange
import random
import os
import math

class DataGenerator:
    def __init__(self, n, camera_function: pd.DataFrame, dirs='generated_images'):
        self.n = n
        self.synth = Synthesizer(camera_function)
        self.dirs = dirs

        print(f'Created initializor for {n} to {self.dirs}')

    def generate_data(self, im_size,template='test', start=0):
        for i in trange(start,self.n+start):
            try:
                self.synth.generate_random_files(im_size[0]*im_size[1], selection_pickle='file_selection_dataframe/success.pkl')
                self.synth.generate_img_from_file(dim=im_size, filename=f'{self.dirs}/{template}_{i}.tiff')
            except ValueError:
                continue
    
    def generate_voronoi(self, im_size, template='voronoi', start=0, min_num_cells=10 ,max_num_cells=20, min_num_materials=5, max_num_materials=10):

        # Add selection data pickle
        self.synth.set_selection_pickle('file_selection_dataframe/success.pkl')
        
        for i in trange(start,self.n+start):
            try:
                rand_num_cells = random.randrange(min_num_cells, max_num_cells)
                rand_num_materials = random.randrange(min_num_materials, max_num_materials)
                fn = f'{self.dirs}/{template}_{i}.tiff'
                self.synth.generate_voronoi(im_size[0], im_size[1], rand_num_cells, rand_num_materials,filename=fn, sample_random=True)
            except ValueError:
                continue

    def generator_sampling(self, im_size, img_type='checkboard', sampling_times=1, template='',start=0, config={}):
        """

        """
        # Add selection data pickle
        self.synth.set_selection_pickle('file_selection_dataframe/success.pkl')
        
        img_folder = template
        if len(template) == 0:
            img_folder = img_type
        
        for i in trange(start, self.n+start):

            # Create full path here
            full_path = os.path.join(self.dirs, f'{img_folder}_{i}')
            os.mkdir(full_path)

            if img_type == 'checkboard':
                # TODO: Complete here, just for fun
                pass
            elif img_type == 'voronoi':
                try:
                    # Get function parameters from config
                    min_num_cells = config['min_num_cells']
                    max_num_cells = config['max_num_cells']
                    min_num_materials = config['min_num_materials']
                    max_num_materials = config['max_num_materials']
                except KeyError as e:
                    print(str(e))

                # Use random to generate the parameter passed into the functions
                rand_num_cells = random.randrange(min_num_cells, max_num_cells)
                rand_num_materials = random.randrange(min_num_materials, max_num_materials)

                for s in range(sampling_times+1):
                    cur_rate = math.pow(2, s)
                    fn = os.path.join(full_path, f'{img_type}_{i}_{int(cur_rate)}x.tiff')
                    # print(f'Sampling down {cur_rate}x...')
                    if cur_rate == 1:
                        try:
                            self.synth.generate_voronoi(im_size[0], im_size[1], rand_num_cells, rand_num_materials,filename=fn, sample_random=True)
                        except ValueError:
                            break
                    else:
                        sample_ratio = 1.0/cur_rate
                        self.synth.sample_img(sample_ratio, save=True, filename=fn)

            else:
                raise ValueError("Img type not recognized, value available: 'chessboard', 'voronoi'")



camera_df = pd.read_pickle('normalized_df.pkl')
dg = DataGenerator(500,camera_df)
dg.generator_sampling((2048,2048), img_type='voronoi', sampling_times=6, config={
    'min_num_cells': 20,
    'max_num_cells': 40,
    'min_num_materials': 10,
    'max_num_materials': 25
})