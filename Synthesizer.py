import pandas as pd
import numpy as np
import utils
from tifffile import imwrite

class Synthesizer:
    def __init__(self, camera_function: pd.DataFrame,
                 start_wavelength=300,
                 end_wavelength=3000,
                 start_threshold=50,
                 end_threshold=500):

        # Camera function
        self.camera_function = camera_function

        # Weighted sum configurations
        self.start_wavelength = start_wavelength
        self.end_wavelength = end_wavelength
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        # Path to pickle serialized dataframe
        self.data_selection_pkl = ''

        # Files array
        self.files_array = np.array([])

        # Image array
        self.img_arr = []

    def generate_random_files(self, n=25, selection_pickle=None):
        """
        Generates random array of size: size from data selection pickle.
        Sets the self.file_array to this array

        :argument n number of element we want to sample
        :arg selection_pickle path of the pickle object that contains a dataframe
        of the filepath we want to use. Should atleast have the column ['file_path']

        :returns: files_array - np array containing random sampled files
        """
        selection_df = pd.DataFrame()

        # If selection_pickle is not None, use the path
        # passed in from the argument and set self.data_selection_pkl
        # to be the argument passed in

        if selection_pickle is not None:
            selection_df = pd.read_pickle(selection_pickle)
            if len(self.data_selection_pkl) == 0:
                self.data_selection_pkl = selection_pickle

        else:
            try:
                selection_df = pd.read_pickle(self.data_selection_pkl)
            except Exception as e:
                print(str(e))

        # Get n samples of the dataframe
        # and return the numpy array
        sampled_df = selection_df.sample(n)

        files_array =  np.array(sampled_df['file_path'])
        self.files_array = files_array

        return files_array

    def save_img(self,filename,ext='.tif'):

        if ext == '.tif':
            imwrite(filename, self.img_arr, planarconfig='CONTIG')


    def generate_img_from_file(self, dim: (int,int), filename: str):
        """
        Doesn't return anything but saves the file under the filename specified
        First we reshape self.files_array and then we find the weighted sum of each element

        :return:
        """
        print('Generating...')
        print(f'Reshaping...\nself.files_array shape:  {self.files_array.shape}')
        reshaped_file_array = np.reshape(self.files_array, dim)
        print(f'Reshaping Done\nself.files_array shape:  {reshaped_file_array.shape}')

        for row in reshaped_file_array:
            new_row = [utils.get_weighted_sums_from_txt_file(x, self.camera_function)[1] for x in row]
            self.img_arr.append(new_row)

        self.save_img(filename)