"""
    By Thomas DeWitt.
    Ising model compatible with cloud_finder.
"""
from warnings import warn
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from numba import njit, prange
from PyThomas import useful_functions as uf

import plotting_functions
from directories import StringDirectories
# base_dir, data_dir, code_files_dir, figures_dir = directories.directories()
dirs = StringDirectories()

def main():
    
    
    # shape = (20000,20000)
    # its = [1,10,50,100,200]
    # potentialits = 80000  
    # for i in its:
    #     ising = Ising(shape, iterations_per_gridpoint=i, starting_probability=0.5)
    #     start = time.time()
    #     ising.create_lattice()

    #     # print(time.time()-start)
    #     # exit()

    #     end = time.time()
    #     print('{:.01f}s for simulation with {} its ({:.01f}h est. for {} ising with {} its)'.format(end-start, i, (potentialits/i)*(end-start)/3600, shape, potentialits))

    shape = (10000,10000)
    its = 0
    isings = [Ising(shape, its, identifier=i) for i in range(10)]

    start = time.time()

    print('Creating Isings')
    for n, i in enumerate(isings):
        i.create_lattice()
        i.save_data()
        print(f'Finished {n+1}/{len(isings)}, {(time.time()-start)/3600:.01f}h elapsed')
    

    print('Total time: {:.0f}s ({:.1f}m or {:.1f}h or {:.1f}d)'.format(time.time()-start,(time.time()-start)/60, (time.time()-start)/3600,(time.time()-start)/(24*3600)))
    

class Ising(uf.SaveLoad):
    """
        Class to house Ising model lattices 
    """
    def __init__(self, shape, iterations_per_gridpoint=1000, starting_probability=0.592746, temperature=2.26918531421, identifier=0):
        """
            Input: 
                shape - 2-tuple
                temperature: float, Temperature in natural units. Critical:  2.26918531421  
                iterations_per_gridpoint: int, Iterations to perform per gridpoint, e.g. if shape is (10,10) and 5 is passed here
                                                every grid point will have 5 opportunities to flip
                starting_probability: float between 0 and 1, the probability to initialize the random initial values as 1
                                                                (passing 0 will initialize with all 0s)
                identifier - unique identifier for saving purposes: int
        """

        # Lattices:
        self.lattice = None     # a binary array of 0s and 1s: 0 not a cloud, 1 is a cloud
        self.edge_clouds_only = None
        # Run parameters:
        self.shape = np.array(shape)
        self.temperature = temperature
        self.iterations_per_gridpoint = iterations_per_gridpoint
        self.starting_probability = starting_probability
        

        self._identity = identifier   # int: index of ising with this shape and other parameters
        # SaveLoad attrs
        self._save_to_path = dirs.ising_dir
        self._is_complete = False        # T/F: whether all data has been found
        self._filename = 'Ising_lattice_{}_{}_{}_{}_{}/'.format(self.shape, self.temperature, self.iterations_per_gridpoint, self.starting_probability, self._identity).replace(' ','_')


    def create_lattice(self):
        """
            Input:
            Output:
                lattice: 2-D binary np.ndarray of shape self.shape

            info on critical temperature: # from https://mattbierbaum.github.io/ising.js/
        """
        try:
            if self.lattice is not None: raise RecursionError('Ising simulation already run')
        except AttributeError:
            pass
    
        if self.starting_probability != 0.5 and self.iterations_per_gridpoint>0:
            warn('Iterating an Ising simulation {} times with a starting probability of {}'.format(self.iterations_per_gridpoint, self.starting_probability))

        # Generate random lattice to start
        random_number_generator = np.random.default_rng()
        random_values = random_number_generator.random(size=self.shape)
        lattice = np.zeros(self.shape, dtype=np.int8)
        lattice[random_values<self.starting_probability] = 1
        lattice[lattice==0] = -1
        # Create checkerboard. For each iteration can only change checkerboarded locations.
        odd_indices = np.zeros(self.shape,dtype=bool)
        even_indices = np.zeros(self.shape,dtype=bool)
        odd_indices[1::2,1::2] = 1
        even_indices[0::2,0::2] = 1

        checkerboard = odd_indices.copy()
        checkerboard[:,0::2] = even_indices[:,0::2]

        lattice = self._iterate_lattice(lattice, checkerboard, random_number_generator)
        
        lattice[lattice==-1] = 0    # convert to cloud/not cloud
        self.lattice = lattice.astype(bool)
        self._is_complete = True

    def show_lattice(self):
        fig, ax = plt.subplots(1,1, subplot_kw={ 'aspect': 'equal'}, figsize=(14,8))
        ax.pcolormesh(self.lattice, cmap = plotting_functions.cloud_colors)
        plt.title('Shape: {}, Iterations: {}, Starting P: {}, Temp: {:.01f}'.format(self.shape, self.iterations_per_gridpoint, self.starting_probability, self.temperature))
        plt.show()

    def _iterate_lattice(self, lattice, checkerboard, random_number_generator):

        for _ in range(self.iterations_per_gridpoint):
            # If flipping the location lowers the energy, do, it, otherwise
            # flip with probability of the Boltzmann factor. 
            # Can do this with every other value at once.
            random_numbers = random_number_generator.random(size=lattice.shape)
            lattice = _iterate_lattice_helper(lattice, checkerboard, random_numbers, self.temperature)
            lattice = _iterate_lattice_helper(lattice, ~checkerboard, random_numbers, self.temperature)
        return lattice

@njit(parallel=True)
def _iterate_lattice_helper(lattice, checkerboard, random_numbers, temperature):

    locations = np.argwhere(checkerboard==1)
    next_lattice = lattice.copy()
    for loc in prange(len(locations)):
        i, j = locations[loc]

        if i == lattice.shape[0]-1: top = lattice[0,j] 
        else: top = lattice[i+1, j]
        if i == 0: bottom = lattice[lattice.shape[0]-1, j]
        else: bottom = lattice[i-1,j]

        if j == lattice.shape[1]-1: right = lattice[i, 0]
        else: right = lattice[i, j+1]
        if j == 0: left = lattice[i, lattice.shape[1]-1]
        else: left = lattice[i, j-1]

        delta_energy = lattice[i,j]*2*(top+bottom+left+right)

        if delta_energy<=0: 
            next_lattice[i,j] = lattice[i,j] * -1   # flip if lowers energy, otherwise flip with probability of Bolzmann factor
            continue

        if temperature == 0: boltzmann_factor = 0
        else: boltzmann_factor = np.exp(-delta_energy/temperature)

        if random_numbers[i,j]<boltzmann_factor: next_lattice[i,j] = lattice[i,j] * -1

    return next_lattice

def multiprocess_Ising(isings, n_processes=4):
    """
        Input:
            isings: List of Ising objects that have not had properties calculated yet: list
            n_processes: int
        Output:
            List of Ising objects with complete properties

        Multiprocess the Ising.find_all_properties() function.
    """
    n_processes = min([n_processes,len(isings)])
    if n_processes == 1:
        [*map(_multiprocess_Ising_helper, isings)]
    else:
        with Pool(n_processes) as p:
            isings = p.map(_multiprocess_Ising_helper, isings)
    for ising in isings: ising.save_or_load_data()
    return isings

def _multiprocess_Ising_helper(ising):
    # Run find_all_properties() on an Ising instance
    ising.find_all_properties()
    return ising

def divide_ising(ising, number_splits_x, number_splits_y, save=True):
    """
        Input:
            ising: Ising object with ising.lattice != None
            number_splits_x: number of sub-lattices to make in x direction
            number_splits_y: number of sub-lattices to make in y direction
            save: whether to save the resulting isings
        Output:
            list of Ising objects with special attribute:
                -original_location: the index i,j of where the lattice was taken from the parent
                -Ising.filename is modified version of parent's filename

        Given an Ising model with lattice, return a list of Isings with smaller lattices taken from the single larger one.
        Calculate all parameters and (optional) save each lattice.
    """
    if ising.shape[1]%number_splits_x != 0: raise ValueError('number_splits_x must be factor of ising.shape[1]')
    if ising.shape[0]%number_splits_y != 0: raise ValueError('number_splits_y must be factor of ising.shape[0]')

    new_isings = []
    for i, narrow_lattice in enumerate(np.split(ising.lattice, number_splits_x, axis=1)):
        for j, new_lattice in enumerate(np.split(narrow_lattice, number_splits_y, axis=0)):
            new_ising = Ising(new_lattice, ising.iterations_per_gridpoint, ising.starting_probability, ising.temperature, ising._identity)
            new_ising.lattice = new_lattice
            new_ising.original_location = (i,j)
            new_ising._filename = 'Sub_'+new_ising._filename+'{}_{}'.format(i,j)
            new_ising.find_all_properties(create_lattice=False)
            if save: new_ising.save_data()
            new_isings.append(new_ising)
    return new_isings

def determine_iterations_for_fractal_D(desired_D, desired_accuracy=.01, shape=(1000,1000), n_processes=8, show_plots=False):
    """
        Start at 0 temperature, random state. Iterate some N_iterations, see if fractal dimension is close, 
        use that to determine next value of N_iterations until D is within +/- desired_accuracy.
        Return N_iterations per pixel.

        For large N_iterations, D = 1
        For small N_iterations, D = 1.7 ish
    """
    N_iterations = 10*shape[0]*shape[1]
    D=0

    counter = 0
    while (D<desired_D-desired_accuracy) or (D>desired_D+desired_accuracy):


        lattice = create_lattice((False, None, (shape, 0, N_iterations, 'wrap')))[0]

        D, error = cf.determine_fractal_D_fast(lattice, n_processes=n_processes, show_plots=False)

        if show_plots:
            plt.pcolormesh(lattice)
            plt.show()

        delta_iteration = np.abs(1000*np.mean(shape)*(desired_D-D))
        if D > desired_D:
            N_iterations = int(N_iterations+delta_iteration)
        elif D < desired_D:
            N_iterations = int(N_iterations-delta_iteration)
        
        if N_iterations<0: N_iterations = 0
        print('Distance to desired D: {}'.format(round(desired_D-D, 2)))
        counter+=1

    print('Took {} tries'.format(counter))
    return N_iterations/(shape[0]*shape[1])

def plot_lattice(lattice):
    fig, ax = plt.subplots(1,1, subplot_kw={ 'aspect': 'equal'}, figsize=(14,8))
    ax.pcolormesh(lattice)
    plt.show()

if __name__ == '__main__':
    main()