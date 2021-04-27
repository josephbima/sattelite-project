import threading
import time
from DataGenerator import DataGenerator
import pandas as pd

CONFIG ={
        'start': 0,
        'num_images': 500,
        'im_size': (2048,2048),
        'img_type': 'voronoi',
        'sampling_times': 6,
        'min_num_cells': 20,
        'max_num_cells': 40,
        'min_num_materials': 10,
        'max_num_materials': 25
}

class myThread(threading.Thread):
    def __init__(self, threadID, name, CONFIG ={
        'start': 0,
        'num_images': 500,
        'im_size': (2048,2048),
        'img_type': 'voronoi',
        'sampling_times': 6,
        'min_num_cells': 20,
        'max_num_cells': 40,
        'min_num_materials': 10,
        'max_num_materials': 25
    }):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.camera_df = pd.read_pickle('normalized_df.pkl')
        self.config = CONFIG

    def run(self):
      print(f"Starting {self.name}")
      generate_voronoi_sample(self.camera_df, self.config['num_images'], self.config['im_size'], self.config['img_type'], self.config['sampling_times'], self.config, self.config['start'])
      print(f"Exiting {self.name}")


def generate_voronoi_sample(camera_df, num_images,im_size, img_type ,sampling_times, config, start=0):
    dg = DataGenerator(num_images, camera_df)
    dg.generator_sampling(im_size=im_size, img_type=img_type, sampling_times=sampling_times, start=start, config=config)


CONFIG['num_images'] = 10
thread1 = myThread(1, "T1", CONFIG=CONFIG )

CONFIG2 = CONFIG.copy()
CONFIG2['start'] = 10
thread2 = myThread(2, "T2", CONFIG=CONFIG2 )

# Start new Threads
thread1.start()
thread2.start()

print("Exiting Main Thread")