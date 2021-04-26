import pandas as pd
import numpy as np
from random import choices

# This will make sure the dataframe returned is not erronous
def read_df_from_txt(file_path):
    df = pd.read_table(file_path, sep='\t', skiprows=2, names=['wavelength', 'reflectance', 'std'])

    #     pd.to_numeric(df['wavelength'], errors='coerce', downcast='float')
    #     pd.to_numeric(df['reflectance'], errors='coerce', downcast='float')
    #     pd.to_numeric(df['std'], errors='coerce', downcast='float')

    for index, row in df.iterrows():
        try:
            row['wavelength'] = float(row['wavelength'])
            row['reflectance'] = float(row['reflectance'])
            row['std'] = float(row['std'])
        except Exception:
            #             print(f'Dropped row: \n{row} \n')
            df.drop(index, inplace=True)

    # df.astype('int64').dtypes

    return df


def resample(df:pd.DataFrame, n=5):
    """
    Resample data to be n steps
    """
    df2 = df.groupby(df.index // n).mean()
    df2.index *= n
    return df2


# Preprocess wavelength data
# Resampling the wavelength into steps of n
# Investigate weird errors
def preprocess_material_response(df, n=5, start=300, limit=3000, start_threshold=50, end_threshold=500,
                                 ignore_limits=False):
    # Assuming the format is df" [wavelength, reflectance]
    
    # We first convert the wavelength from micron to nanometer
    df['wavelength'] = df['wavelength'].apply(lambda x: int(x * 1000))

    # Group by wavelength to use wavelength as index
    df = df.groupby('wavelength').mean()

    # Here we make sure that the steps of the wevelength are of step n
    df2 = resample(df, n)

    # To reduce runtime, we can limit the range we are evaluating to
    # This limit should be set so it is the same as the limit of the camera response function
    # We want to make sure that the start and end is within the boundaries that we set

    ori_fvi = df2.first_valid_index()
    ori_lvi = df2.last_valid_index()

    df2 = df2.loc[df2.index >= start]
    df2 = df2.loc[df2.index <= limit]

    if ignore_limits == False:
        fvi = df2.first_valid_index()
        lvi = df2.last_valid_index()

        if fvi is None or lvi is None:
            raise ValueError(
                f'The start ({start}) and limit ({limit}) values yielded an empty dataframe. The material wavelength started at {ori_fvi} and ends at {ori_lvi}')

        if abs(fvi - start) > start_threshold:
            raise ValueError(
                f'The material wavelength starts at {fvi} while our starting value is {start} with a threshold of {start_threshold}')

        if abs(lvi - limit) > end_threshold:
            raise ValueError(
                f'The material wavelength ends at {lvi} while our ending value is {limit} with a threshold of {end_threshold}')

    return df2


def get_weighted_sum(material_df, camera_df, band_name):
    weight_sum = 0
    missing_idx = set()

    for wv, row in material_df.iterrows():
        try:
            if camera_df.at[wv, band_name] > 0:
                weight_sum += (row['reflectance'] * camera_df.at[wv, band_name])
        #                 print(f"Wavelength {wv} : {row['reflectance']} * {df1.at[wv, band_name]}")
        except KeyError:
            #            print(f'Key {wv} not found')
            missing_idx.add(wv)
            continue

    #     print(f'Missing wavelength: {missing_idx} with length {len(missing_idx)}')
    return weight_sum, missing_idx


# Gets weighted sum from processed material and camera dataframe

def get_weighted_sums_from_df(material_df, camera_df, highlight_missing=False):
    weighted_sums = []
    for band_name in camera_df.columns:
        wsum, missing = get_weighted_sum(material_df, camera_df, band_name)
        weighted_sums.append(wsum)

    #     print(f'There are {len(missing)} missing wavelengths in the material function')

    # # Normalize the array
    # norm = np.linalg.norm(weighted_sums)
    
    # weighted_sums = weighted_sums / norm

    if highlight_missing:
        print(f'Missing wavelength: {missing} with length {len(missing)}')

    return missing, weighted_sums


def get_weighted_sums_from_txt_file(txt_file, camera_df, config={}):
    df = read_df_from_txt(txt_file)
    df = preprocess_material_response(df,start=config['start_wavelength'],limit=config['end_wavelength'],start_threshold=config['start_threshold'],
                                      end_threshold=config['end_threshold'],ignore_limits=config['ignore_limits'])


    return get_weighted_sums_from_df(df, camera_df)


def get_random_success_files(n=25, pkl='success.pkl'):
    success_df = pd.read_pickle(pkl)
    sampled_df = success_df.sample(n)

    return np.array(sampled_df['file_path'])


def get_random_weighted_sums(n=25, pkl='success.pkl', dim=None):
    """
    Get the weighted sums of n materials from pkl satelite response functions
    @params:
        n   (int)              :  number of random weighted sums to be generated
        pkl (pkl: dataframe)   :  pickle file of camera function in dataframe
        dim ((int,int))        :  dimension of returned array in (height,width)
    """
    camera_df = pd.read_pickle(pkl)
    random_files = get_random_success_files(n, pkl)
    for i, row in enumerate(random_files):
        for j, el in enumerate(row):
            random_files[i][j] = get_weighted_sums_from_txt_file(el, camera_df)

    if dim is not None:
        random_files = np.array(random_files).reshape(dim)

    return random_files


def reshape_array(arr, h, w):
    return np.array(arr).reshape((h, w))

### Growing Algorithm Helper Functions
def find_first_empty_space(grid):
    for row in grid:
        for col in row:
            if grid[row][col] == 0:
                return row,col

    return -1

def get_next_coordinate(grid, start, growth_probability):
    max_row = len(grid)
    max_col = len(grid[0])

    start_row, start_col = start

    # 0 : Means will not grow
    # 1 : Will go right
    # 2 : Will go down
    population = [0,1,2]
    weights = [1-(2*growth_probability), growth_probability, growth_probability]

    next_dir = choices(population, weights)

    if next_dir == 0:
        return -1,-1

    if next_dir == 1:
        if start_col + 1 >= max_col: # If we hit a wall wen moving right
            return -1,-1
        if grid[start_row][start_col+1] != -1: # If we hit an already filled value
            return -1,-1
        
        return start_row, start_col+1

    if next_dir == 2:
        if start_row+1 >= max_row:
            return -1,-1
        if grid[start_row+1][start_col] != -1:
            return -1,-1 
        
        return start_row+1, start_col


    return

def growing_algorithm(grid, value, growth_probability, growth_decay):

    ng = np.array(grid)

    # Find available position
    start = find_first_empty_space(grid)

    # Iterate and grow
    cur_growth = growth_probability
    next_row, next_col = get_next_coordinate(grid, start, growth_probability)

    while next_row != -1 and next_col != -1:
        grid[next_row][next_col] = value
        next_row, next_col = get_next_coordinate(grid, start, growth_probability)

    # Finish
    return grid
