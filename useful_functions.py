"""
    Author: Thomas DeWitt
    General file containing simple functions useful in a wide variety of projects. Sort of a personal package.
"""
import numpy as np
from numba import njit, prange
from numba.typed import List
from scipy.ndimage.measurements import label
from skimage.segmentation import clear_border
import warnings
import os
from directories import StringDirectories
from powerlaw import Power_Law
dirs = StringDirectories()

# Constants
g = 9.8     # m s^-2
L_v = 2264705 # J/kg  # from https://en.wikipedia.org/wiki/Latent_heat#Table_of_specific_latent_heats
L_f = 334000 # J/kg   # from https://en.wikipedia.org/wiki/Latent_heat#Table_of_specific_latent_heats   
c_p = 1004  # J kg^-1 K^-1

# Statistical functions
def bootstrap(data, estimator, n_bootstraps=1000, **kwargs):
    """
        Input:
            data: 1-D series of data to estimate "parameter"
            estimator: function that estimates "parameter" from data. 
                        Should take data as first arg and return
                        (estimate, _) (_ would be error, but is ignored).
                        kwargs are passed to estimator
            n_bootstraps: number bootstraps to perform
            kwargs: passed to estimator

        Output:
            estimate, error
                estimate: equal to estimator(data)[0]
                error: bootstrapped estimate cooresponding to 95\% conf. interval

        Use estimator to estimate an arbitrary parameter describing data using
        bootstrapping.
    """
    main_estimate = estimator(data, **kwargs)
    if type(main_estimate) is not tuple or len(main_estimate) != 2:
        raise ValueError('Function estimator should return (estimate, error)')
    main_estimate = main_estimate[0]

    estimates = []
    for i in range(n_bootstraps):
        estimates.append(estimator(np.random.choice(data, size=len(data)), **kwargs)[0])

    return main_estimate, 2*np.nanstd(estimates)

def estimate_power_law_exponent_lr(data, nbins=30, min_counts=30, min_good_data_order=1):
    """
        Input:
            data: data that are hypothetically power-law distributed
            nbins: number of bins
            min_counts: omit any bin from the regression with fewer counts
            min_good_data_order: number of orders of magnitude over which the good bins must span

        Using the linear regression method, estimate the power-law exponent assuming data follows a distribution
        p(data) \propto data^{-exponent-1}.

        Returns nans if fewer than 3 bins to have counts>min_counts or if good bins span less than min_good_data_order orders of magnitude

        Bin the data into nbins bins, spaced logarithmically between data.min() and data.max(), then perform
        a linear regression to the log(counts) and log(bin_locations)

        Returns: exponent, uncertainty (95% conf. interval)
    """
    data = np.log10(data)
    bin_edges = np.linspace(data.min(), data.max(), nbins+1)
    counts = np.histogram(data, bins=bin_edges)[0].astype(np.float32)
    counts[counts<min_counts] = np.nan

    # Test size of good data
    if np.count_nonzero(np.isfinite(counts)) < 3: 
        return np.nan, np.nan
    if max(bin_edges[:-1][np.isfinite(counts)]) - min(bin_edges[:-1][np.isfinite(counts)]) < min_good_data_order: 
        return np.nan, np.nan

    (slope, _), (error, _) = linear_regression(bin_edges[:-1], np.log10(counts))

    return -slope, 2*error

def estimate_truncated_power_law_exponent_MLE(data, alphamin=0.5, alphamax=3, dalpha=.01):
    """
        Use a numerical algorithm to maximize the likelihood function that data were drawn 
        from a power law distribution with alpha between alphamin and alphamax, i.e.

        n(data) \propto data^{-alpha-1}

        Numerically maximizes the likelihood function for all alphas between alphamin and alphamax
        with steps dalpha

        See 8/10/23 in research journal, also see Savre 2023.

        Returns estimate, error
        error is dalpha; this is not necessarily a good estimate of the true uncertainty
    """
    data = np.asarray(data)

    potential_alphas = np.arange(alphamin, alphamax, dalpha)

    duplicated_data = np.repeat([data],potential_alphas.size, axis=0)
    duplicated_potential_alphas = np.repeat([potential_alphas], data.size, axis=0).T
    c = duplicated_potential_alphas * (data.min()**(-duplicated_potential_alphas)-data.max()**(-duplicated_potential_alphas))**-1

    individual_likelihoods = np.log10(c * duplicated_data**(-duplicated_potential_alphas))    # taking the log reduces numerical errors because these numbers are tiny
    likelihoods = np.sum(individual_likelihoods, axis=1)
    
    if likelihoods.sum() == 0:
        warnings.warn('Failed to estimate likelihoods')
    estimate_loc = np.argmax(likelihoods)
    if estimate_loc == 0 or estimate_loc == likelihoods.size-1:
        warnings.warn('Truncated power law exponent may be beyond numerical search, change alphamin and alphamax')
    estimate = potential_alphas[estimate_loc]
    error = dalpha 

    return estimate, error

def estimate_unbounded_powerlaw_exponent_MLE(series):
    """
        Estimate the exponent for a power law distribution with no upper bound;
        assume the distribution follows 
        n(data) \propto data^{-alpha-1}

        From Newman 2005 eqn. 5, errors from Clauset 2009

        Return alpha, error for 95\% conf interval

    """

    if len(series[~np.isfinite(series)]) != 0:
        print(f'Dropping {(100*len(series[~np.isfinite(series)])/len(series)):.01f}% of all values in MLE unbounded power law exponent estimate')
    series = series[np.isfinite(series)]
    series = np.sort(series)
    bracket_sum = 0
    for i in series:
        bracket_sum = bracket_sum + np.log(i/series[0])
        
    alpha =  len(series)/bracket_sum
    sigma = (alpha)/np.sqrt(len(series)) # Clauset et al 2009 eqn. 3.2
    
    return alpha, 2* sigma

def generate_truncated_power_law_randoms(n, min_value, max_value, alpha):
    """
    Generate n random variables pulled from a truncated power law between min_value and max_value with exponent alpha, following

        n(x) \propto x^{-alpha-1}
    """

    rands_to_keep = []
    while len(rands_to_keep)<n:
        new_rands = Power_Law(xmin=min_value, parameters=[alpha+1]).generate_random(n-len(rands_to_keep))
        new_rands = new_rands[new_rands<max_value]
        rands_to_keep.extend(new_rands)

    return np.array(rands_to_keep)

def linear_regression(x, y):
    """
        Return (slope, y-int), (error_slope, error_y_int) for 95% conf
    """
    index = np.isfinite(x) & np.isfinite(y)
    if len(x[index]) <3:    # "the number of data points must exceed order to scale the covariance matrix"
        warnings.warn('Less than 3 points (x,y) are good (not nan), returning nans')
        return (np.nan, np.nan),(np.nan, np.nan)
    try:
        coefficients, cov = np.polyfit(x[index], y[index], 1, cov=True)
        error = np.sqrt(np.diag(cov))
    except Exception as e:
        warnings.warn('Linear regression failed, error message\n','     ',e)
        return (np.nan, np.nan),(np.nan, np.nan)
    return coefficients, 2*error  # 95% conf interval is 2 times standard error 

# 1D and simpler functions
def chunk_list(original_list, n_per_chunk, allow_smaller_chunk=False):
    """
    Divide the original list into chunks of size n_per_chunk.

    Parameters:
    original_list (list): The list to be divided into chunks.
    n_per_chunk (int): The size of each chunk.
    allow_smaller_chunk (bool, optional): If True, the last chunk may be smaller than n_per_chunk. If False, the last chunk will be of size n_per_chunk or omitted if smaller.

    Returns:
    list: A list of chunks, where each chunk is a sublist of original_list.

    Examples:
    >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, allow_smaller_chunk=True)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, allow_smaller_chunk=False)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    Note:    if allow_smaller_chunk is True, sometimes an empty list is included at the end if the list length does divide evenly by n_per_chunk.
    """
    if allow_smaller_chunk:
        chunks = [original_list[i*n_per_chunk:(i+1)*n_per_chunk] for i in range(0, len(original_list)//n_per_chunk+1) ]
    else:
        chunks = [original_list[i*n_per_chunk:(i+1)*n_per_chunk] for i in range(0, len(original_list)//n_per_chunk+1) if len(original_list[i*n_per_chunk:(i+1)*n_per_chunk]) == n_per_chunk ]
    return chunks

def bin_and_sort(bin_edges, binned, taken_with, return_means = False):
    """
        Sort 'binned' into 'bin_edges'. Sort taken_with into the bins that binned were put into. taken_with is a list of lists, each of which is sorted.

        Sort each list in taken_with into bins. Each element of each list in taken_with is sorted into the bin that the corresponding value in binned gets sorted into.

        binned: 1-D: thing to be binned into bin_edges
        taken_with: list of 1-D, each of len binned. If taken_with is only 1-D (still of len(binned)), the first index of sorted_taken_with may be ignored and is of len 1.

        Returns binned_counts, sorted_taken_with
        where sorted_taken_with is a list of lists of lists: first index corresponding to which of taken_with, second which bin, third is the sorted values.

        if return_means:
        Returns binned_counts, sorted_taken_with, sorted_taken_with_means
    """
    if len(binned) == len(taken_with) and type(taken_with[0]) != list: 
        taken_with = [taken_with]
    for num, i in enumerate(taken_with): 
        if len(binned) != len(i): raise ValueError(f'Element {num} of taken_with is not of same len as binned. Element is {i}')

    # Following 2 lines are equivalent to  (but we need indices)
    # counts = np.histogram(binned, bins=bin_edges)[0]
    indices = np.digitize(binned, bin_edges)    # area
    binned_counts = np.bincount(indices, minlength=len(bin_edges)+1)[1:-1].astype(float)

    sorted_taken_with = []
    sorted_taken_with_means = []

    for thing_to_sort in taken_with:
        average_things = []
        binned_things = []
        for i in range(len(bin_edges)-1):
            thing = np.compress(indices==i+1, thing_to_sort)

            binned_things.append(thing)
            if return_means:
                if len(thing) > 0:
                    average_things.append(np.mean(thing))
                else: average_things.append(np.nan)
        sorted_taken_with.append(binned_things)
        sorted_taken_with_means.append(average_things)
    
    if return_means: return  binned_counts, sorted_taken_with, np.squeeze(sorted_taken_with_means)
    else: return  binned_counts, sorted_taken_with

# Simple 2D array functions
def upscale_array(array, factor, make_binary=True):
    """
    Upscale a given array by a specified factor along both dimensions.

    This function takes an input array and reduces it by a given factor along both the x and y dimensions. The
    upscaling is achieved by summing 'superpixel' regions of the original array and dividing by the square of the
    scaling factor. Optionally, the resulting upscaled array can be thresholded to create a binary array, where
    values above 0.5 are set to 1 and values below or equal to 0.5 are set to 0.

    Args:
        array (numpy.ndarray): The input array to be upscaled.
        factor (int): The scaling factor for enlarging the array. Must be a positive integer.
        make_binary (bool, optional): If True, the resulting upscaled array will be thresholded to create a binary
            array. Values above 0.5 will be set to 1, and values below or equal to 0.5 will be set to 0. Defaults to True.

    Returns:
        numpy.ndarray: The upscaled array.

    Raises:
        ValueError: If an even scaling factor is provided while attempting to create a binary array, as this may lead
            to rounding issues.

    Note:
        For odd scaling factors, there will not be places in the upscaled array where the value is exactly 0.5.

    Example:
        original_array = np.array([[1, 2],
                                    [3, 4]])
        upscaled = upscale_array(original_array, factor=2, make_binary=False)
        # Resulting upscaled array:
        # array([2.5])

    """
    if (factor // 2 == factor/2) and make_binary: raise ValueError('Only odd factors currently supported for binary matrices')   # this is because of rounding issues
    # if array.shape[0]//factor != array.shape[0]/factor: raise ValueError('array must be evenly divisible by factor')
    # if array.shape[1]//factor != array.shape[1]/factor: raise ValueError('array must be evenly divisible by factor')

    upscaled_array = np.add.reduceat(array, np.arange(array.shape[0], step=factor), axis=0)     # add points in x direction
    upscaled_array = np.add.reduceat(upscaled_array, np.arange(array.shape[1], step=factor), axis=1)     # add points in y direction

    upscaled_array = upscaled_array/factor**2
    if make_binary:
        upscaled_array[upscaled_array >.5] = 1
        upscaled_array[upscaled_array <.5] = 0  # for odd factor, there will not be places where it =.5

    return upscaled_array


def get_structure_props(array, x_sizes, y_sizes, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), print_none=False, wrap=None):
    """
        Input:
            array - Binary array of strc: 2-d np.ndarray, padded with 0's or np.nan's
            x_sizes = Sizes of pixels in horizontal direction, same shape as array: 2-d np.ndarray
            y_sizes = Sizes of pixels in vertical direction, same shape as array: 2-d np.ndarray
            structure = Defines connectivity
            print_none = Print message if no structures found
            wrap = None, 'sides, 'all: 
                if 'sides', connect structures that span the left/right edge
                if 'both', connect structures that span the left/right edge and top/bottom edge
        Output:
            perimeter, area, height, width: 1-D np.ndarrays, each element the perimeter/area/height/width of an individual structure

        Note: if x_sizes or y_sizes are not uniform, the width will be the sum of the average pixel widths of the pixels in the column and in the object.
        Similarly, the height will be the sum of the average pixel heights of the pixels in the row and in the object.
        Given a array and the sizes of each pixel in each direction, calculate properties of structures. 
        Any perimeter between structure and nan is not counted. 
    """

    if array.shape != x_sizes.shape or array.shape != y_sizes.shape: 
        raise ValueError('array, x_sizes, and y_sizes must all be same shape. Currently {},{},{}'.format(array.shape, x_sizes.shape, y_sizes.shape))

    if np.count_nonzero((np.isnan(x_sizes) | np.isnan(y_sizes)) & np.isfinite(array)): 
        raise ValueError('x or y sizes are nan in locations where array is not')
    
    # if 1 in array[0] or 1 in array[:,0] or 1 in array[-1] or 1 in array[:,-1]: raise ValueError('array must be padded with 0s or nans.')
    no_nans = array.copy()
    no_nans[np.isnan(array)] = 0   # so we don't consider nans structures and also so they don't connect multiple structures
    if np.count_nonzero(no_nans) == 0: 
        if print_none: print('No structures found')
        return np.array([]),np.array([]),np.array([]),np.array([])
    labelled_array, n_structures = label(no_nans.astype(bool), structure, output=np.float32)      # creates array where every unique structure is composed of a unique number, 1 to n_structures

    if wrap is None: pass
    elif wrap == 'sides':
        # set those on right to the same i.d. as those on left
        for j,value in enumerate(labelled_array[:,0]):
            if value != 0:
                if labelled_array[j, labelled_array.shape[1]-1] != 0 and labelled_array[j, labelled_array.shape[1]-1] != value:
                    # want not a structure and not already changed
                    labelled_array[labelled_array == labelled_array[j, labelled_array.shape[1]-1]] = value  # set to same identification number
    
    if wrap is None: pass
    elif wrap == 'both' or wrap == 'sides': 
        labelled_array = label_periodic_boundaries(labelled_array, wrap)
    else: raise ValueError(f'wrap={wrap} not supported')

    # Flatten arrays to find their indices.
    values = np.sort(labelled_array.flatten())
    original_locations = np.argsort(labelled_array.flatten())  # Get indices where the original values were
    indices_2d = np.array(np.unravel_index(original_locations, labelled_array.shape)).T    # convert flattened indices to 2-d
    
    labelled_array[np.isnan(array)] = np.nan      # Turn this back to nan so perimeter along it is not included
    split_here = np.roll(values, shift=-1)-values   # Split where the values changed.
    split_here[-1] = 0                  # Last value rolled over from first

    separated_structure_indices = np.split(indices_2d, np.where(split_here!=0)[0]+1)
    separated_structure_indices = separated_structure_indices[1:]   # Remove the locations that were 0 (not structure)
    if len(separated_structure_indices) == 0: return np.array([]),np.array([]),np.array([]),np.array([])

    # x_coords, y_coords to calculate width/height
    # x_coords = np.nancumsum(x_sizes, 1)
    # y_coords = np.nancumsum(y_sizes, 0)


    # must use numba.typed.List here for some reason https://numba.readthedocs.io/en/stable/reference/pysupported.html#feature-typed-list   
    p, a, h, w = _get_structure_props_helper(labelled_array, List(separated_structure_indices), x_sizes, y_sizes) 
    nanmask = np.logical_or(np.logical_or(np.isnan(p), np.isnan(a)), np.logical_or(np.isnan(h), np.isnan(w)))
    if np.count_nonzero(nanmask) > 0: raise ValueError('Nan values found: {} out of {}'.format(np.count_nonzero(nanmask), len(p)))
    p, a, h, w = np.array(p), np.array(a), np.array(h), np.array(w)
    p, a, h, w = p[~nanmask], a[~nanmask],h[~nanmask], w[~nanmask]
    return p,a,h,w

@njit()
def _get_structure_props_helper(labelled_array, separated_structure_indices, x_sizes, y_sizes):
    
    p, a, = [],[]
    h, w = [],[]

    for indices in separated_structure_indices:
        perimeter = 0
        area = 0

        y_coords_structure = np.array([c[0] for c in indices])
        x_coords_structure = np.array([c[1] for c in indices])
        unique_y_coords = []
        unique_x_coords = []
        height = 0
        width = 0

        for (i,j) in indices:
            # Height, Width
            if i not in unique_y_coords:
                unique_y_coords.append(i)
                indices = (y_coords_structure==i)
                y_sizes_here = []
                for loc,take in enumerate(indices):
                    if take: y_sizes_here.append(y_sizes[y_coords_structure[loc],x_coords_structure[loc]])
                y_sizes_here = np.array(y_sizes_here)
                height += np.mean(y_sizes_here)
            if j not in unique_x_coords:
                unique_x_coords.append(j)
                indices = (x_coords_structure==j)
                x_sizes_here = []
                for loc,take in enumerate(indices):
                    if take: x_sizes_here.append(x_sizes[y_coords_structure[loc],x_coords_structure[loc]])
                x_sizes_here = np.array(x_sizes_here)
                width += np.mean(x_sizes_here)

            # Perimeter:
            if i != labelled_array.shape[0]-1 and labelled_array[i+1, j] == 0: perimeter += x_sizes[i,j]
            elif i == labelled_array.shape[0]-1 and labelled_array[0, j] == 0: perimeter += x_sizes[i,j]

            if i != 0 and labelled_array[i-1, j] == 0: perimeter += x_sizes[i,j]
            elif i == 0 and labelled_array[labelled_array.shape[0]-1, j] == 0: perimeter += x_sizes[i,j]

            if j != labelled_array.shape[1]-1 and labelled_array[i, j+1] == 0: perimeter += y_sizes[i,j]
            elif j == labelled_array.shape[1]-1 and labelled_array[i, 0] == 0: perimeter += y_sizes[i,j]

            if j != 0 and labelled_array[i, j-1] == 0: perimeter += y_sizes[i,j]
            elif j == 0 and labelled_array[i, 0] == 0: perimeter += y_sizes[i,j]

            # Area:
            area += y_sizes[i,j] * x_sizes[i,j]


        if area != 0: 
            p.append(perimeter)
            a.append(area)
            h.append(height)
            w.append(width)


    return p, a, h, w

def label_periodic_boundaries(labelled_array, wrap):
    if wrap == 'sides' or wrap == 'both':
        # set those on right to the same i.d. as those on left
        for j,value in enumerate(labelled_array[:,0]):
            if value != 0:
                if labelled_array[j, labelled_array.shape[1]-1] != 0 and labelled_array[j, labelled_array.shape[1]-1] != value:
                    # want not a structure and not already changed
                    labelled_array[labelled_array == labelled_array[j, labelled_array.shape[1]-1]] = value  # set to same identification number
    
    if wrap == 'both': 
        # set those on top to the same i.d. as those on bottom
        for i,value in enumerate(labelled_array[0,:]):
            if value != 0:
                if labelled_array[labelled_array.shape[0]-1,i] != 0 and labelled_array[labelled_array.shape[0]-1,i] != value:
                    # want not a structure and not already changed
                    labelled_array[labelled_array == labelled_array[labelled_array.shape[0]-1,i]] = value  # set to same identification number
    return labelled_array

def clear_border_adjacent(array, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
    """
        Input:
            array: 2-D np.ndarray consisting of 0s and 1s 
        Output:
            2-D np.ndarray consisting of 0s and 1s with border structures removed

        Remove connected regions that touch the edge, using a connectivity determined 
        by structure. Similar to skimage.segmentation.clear_border but structure
        can be changed.

            Examples:
                    [[0,0,0,0],                [[0,0,0,0],                [[0,0,0,0], 
                    [0,1,1,0],                 [0,1,1,0],                 [0,1,0,0], 
                    [0,0,0,1],                 [0,0,1,1],                 [1,0,0,0],
                    [0,0,0,0]]                 [0,0,0,0]]                 [0,0,0,0]]
            so ex 1 and 3 would still have one cloud in output but ex 2 would have 0
            for a structure of np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).
    """
    border_cleared = clear_border(label(array.astype(bool), structure)[0])
    border_cleared[border_cleared > 0] = 1
    return border_cleared.astype(bool)

def clear_one_border(array, which):
    """
        Input:
            array: 2-D np.ndarray consisting of 0s and 1s 
            which: 'top','bottom','left','right'
        Output:
            2-D np.ndarray consisting of 0s and 1s with border structures removed

        Remove connected regions that touch the specified edge, using a connectivity of 
        adjacent only. Similar to skimage.segmentation.clear_border but does not
        consider diagonal 1's to be connected.

        Similar to clear_border_adjacent; but only clears one edge
    """
    # Append 1's to the side we want to clear. Label array. Identify which label
    # the appeneded 1's became, then set all of that label to zero.
    # Set nonzero values (remaining clouds) to 1 and return.
    if which == 'bottom':
        with_ones = np.append(np.ones((1, array.shape[1])), array, axis=0)
        with_ones_labelled,_ = label(with_ones)
        cleared = np.copy(with_ones)
        cleared[with_ones_labelled == with_ones_labelled[0,0]] = 0
        return np.delete(cleared, 0, axis=0).astype(bool)
    elif which == 'left':
        with_ones = np.append(np.ones((array.shape[0], 1)), array, axis=1)
        with_ones_labelled,_ = label(with_ones)
        cleared = np.copy(with_ones)
        cleared[with_ones_labelled == with_ones_labelled[0,0]] = 0
        return np.delete(cleared, 0, axis=1).astype(bool)
    elif which == 'right':
        with_ones = np.append(array, np.ones((array.shape[0], 1)), axis=1)
        with_ones_labelled,_ = label(with_ones)
        cleared = np.copy(with_ones)
        cleared[with_ones_labelled == with_ones_labelled[with_ones_labelled.shape[0]-1, with_ones_labelled.shape[1]-1]] = 0
        return np.delete(cleared, -1, axis=1).astype(bool)
    elif which == 'top':
        with_ones = np.append(array, np.ones((1, array.shape[1])), axis=0)
        with_ones_labelled,_ = label(with_ones)
        cleared = np.copy(with_ones)
        cleared[with_ones_labelled == with_ones_labelled[with_ones_labelled.shape[0]-1, with_ones_labelled.shape[1]-1]] = 0
        return np.delete(cleared, -1, axis=0).astype(bool)
    else:
        raise ValueError('Value for which not accepted: "{}"'.format(which))

def encase_in_value(array, value=np.nan, dtype=np.float32):
    """
        Input:
            array: 2-D np.ndarray
            value: value to append on the edge
            dtype: dtype of the resulting array
        Output:
            array: Same as input but with a layer of value all around the edge: 2-D np.ndarray
    """

    nans_lr = np.empty((array.shape[0],1), dtype=dtype)
    nans_tb = np.empty((1, array.shape[1]+2), dtype=dtype)  # will be two bigger after first appends
    nans_lr[:], nans_tb[:] = value, value
    array = np.append(nans_lr, array, axis=1)
    array = np.append(array, nans_lr, axis=1)
    array = np.append(nans_tb, array, axis=0)
    array = np.append(array, nans_tb, axis=0)
    return array

def split_array(parent_array, div_x, div_y):
    """
    Divide a single array into smaller sub-arrays by specified divisions.

    This function takes an input array and divides it into smaller sub-arrays
    by performing splits along the specified dimensions. It returns a list containing
    the resulting sub-arrays.

    Args:
        parent_array (numpy.ndarray): The input array to be divided.
        div_x (int): Number of divisions along the x-axis (columns) for the array.
        div_y (int): Number of divisions along the y-axis (rows) for each sub-array
                     resulting from the x-axis divisions.

    Returns:
        list of numpy.ndarray: A list containing the divided and reduced sub-arrays.

    Note:
        - This function requires the `numpy` library to be imported.
        - The dimensions of the input array should be compatible with the specified divisions.

    Example:
        parent_array = array([[1,  2,  3,  4],
                              [5,  6,  7,  8],
                              [9,  10, 11, 12],
                              [13, 14, 15, 16]])
        div_x = 2
        div_y = 2
        result = split_array(parent_array, div_x, div_y)
        for r in result: print(r)
            [[1 2]
            [5 6]]
            [[ 9 10]
            [13 14]]
            [[3 4]
            [7 8]]
            [[11 12]
            [15 16]]
    """
    new_arrays = []
    for i, narrow_array in enumerate(np.split(parent_array, int(div_x), axis=1)):
        for j, new_array in enumerate(np.split(narrow_array, int(div_y), axis=0)):
            new_arrays.append(new_array)
    return new_arrays

def prune_nans(array):
    """
        Remove rows or columns that are 1) entirely nans and 2) on the exterior of the array

        Args:
        array (np.ndarray): 2D NumPy array containing numeric values and np.nan.

        Returns:
        np.ndarray: A new 2D NumPy array with rows and columns entirely filled with np.nan removed.

        Example:
        array = [[np.nan,0,0,1,0,0,0,np.nan,np.nan]]
                [[np.nan,0,9,1,0,8,0,np.nan,np.nan]]
                [[np.nan,0,0,1,0,0,0,np.nan,np.nan]]
                [[np.nan,0,0,1,1,0,0,np.nan,np.nan]]
                [[np.nan,0,0,1,0,0,0,np.nan,np.nan]]
                [[np.nan,0,0,1,0,1,0,np.nan,np.nan]]
                [[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]
                [[np.nan,0,0,1,0,0,0,np.nan,np.nan]]
                [[np.nan,0,0,1,0,4,0,np.nan,np.nan]]
                [[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]
                [[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]

        prune_nans(array) = [[0,0,1,0,0,0]]
                            [[0,9,1,0,8,0]]
                            [[0,0,1,0,0,0]]
                            [[0,0,1,1,0,0]]
                            [[0,0,1,0,0,0]]
                            [[0,0,1,0,1,0]]
                            [[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]]
                            [[0,0,1,0,0,0]]
                            [[0,0,1,0,4,0]]
    """

    # Find rows and columns with all np.nan values
    nan_rows = np.all(np.isnan(array), axis=1)
    nan_columns = np.all(np.isnan(array), axis=0)
    
    # Find the indices of the first and last non-nan rows and columns
    first_non_nan_row = np.argmax(~nan_rows)
    last_non_nan_row = len(nan_rows) - 1 - np.argmax(np.flip(~nan_rows))
    first_non_nan_column = np.argmax(~nan_columns)
    last_non_nan_column = len(nan_columns) - 1 - np.argmax(np.flip(~nan_columns))
    
    # Extract the non-nan rows and columns from the original array
    pruned_array = array[first_non_nan_row:last_non_nan_row + 1, first_non_nan_column:last_non_nan_column + 1]
    
    return pruned_array

def distance_from_loc(shape, point_to_calc_dist='center', boundaries='inside'):
    """
        Return an array where every value is the distance from the point "point_to_calc_dist". Shape must be odd.
        point_to_calc_dist: pair of indices or "center". If "center", shape must be odd.
        boundaries: if 'inside', distance is the shortest line between points.
                    if 'wrap', the array is assumed periodic: distance is shorter than 'inside' for pairs of points near opposite edges.
    """
    if len(shape)!=2: raise ValueError('only 2 dimensional shapes allowed')
    if point_to_calc_dist == 'center':
        if shape[0] %2 != 1: raise ValueError('shape must be odd')
        if shape[1] %2 != 1: raise ValueError('shape must be odd')

        point_to_calc_dist = (int(shape[0]/2), int(shape[1]/2))
    else:
        if (type(point_to_calc_dist) != tuple) or (point_to_calc_dist[0]>=shape[0]) or (point_to_calc_dist[1] >= shape[1]):
            raise ValueError('point_to_calc_dist must be of type tuple with values strictly smaller than shape')


    if boundaries == 'inside':
        distances_x, distances_y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
        distances_x = distances_x - distances_x[point_to_calc_dist]
        distances_y = distances_y - distances_y[point_to_calc_dist]
    elif boundaries == 'wrap':
        distances_x, distances_y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
        distances_x = np.minimum(np.abs(distances_x - distances_x[point_to_calc_dist]), shape[1] - np.abs(distances_x - distances_x[point_to_calc_dist]))
        distances_y = np.minimum(np.abs(distances_y - distances_y[point_to_calc_dist]), shape[0] - np.abs(distances_y - distances_y[point_to_calc_dist]))
    else: raise ValueError(f'boundaries={boundaries} not supported')

    return np.sqrt(distances_x**2 + distances_y**2)

def make_gaussian_bump(shape, char_dist=10):
    bump =  np.exp(-(distance_from_loc(shape, point_to_calc_dist=(int(shape[0]/2),int(shape[1]/2)))/char_dist)**2)
    return bump

# Simple 3D array functions
def get_3D_structure_widths_heights(array, y_sizes, z_sizes, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), wrap='sides'):
    """
        Given a 3-D cloud mask (array), loop through 1st index (x) to calculate 2-D structure 
        widths, heights, areas, and perimeters for every vertical slice in array

        array: 3D np.ndarray, axes are (x,y,z)
        x_sizes: 3D np.ndarray of shape array.shape
        z_sizes: 3D np.ndarray of shape array.shape

        Return p,a,h,w: np.ndarrays
    """
    p,a,h,w = [],[],[],[]

    for x_ind in range(array.shape[0]):
        # Need to transpose matrices so that first axis is z and second is y, so that it is ``upright''
        new_p,new_a,new_h,new_w = get_structure_props(array[x_ind].T, y_sizes[x_ind].T, z_sizes[x_ind].T, structure, wrap=wrap)
        
        p.extend(new_p)
        a.extend(new_a)
        h.extend(new_h)
        w.extend(new_w)

    return np.array(p), np.array(a), np.array(h), np.array(w)

# More complicated image functions
def array_size_distribution(array, variable='area', bins=30, bin_logs=True, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), wrap=None, x_sizes=None, y_sizes=None):
    """
        Given a binary array, calculate contiguous object sizes and bin them by area/perimeter/length/width

        Input:
            - array: 2-D np.ndarray, where objects of interest have value 1, the background has value 0, and no data has np.nan. 
                        Nans are treated like 0's, except the perimeter along them is not counted.
            - variable: 'area','perimeter','height','width': which object attribute to bin by
            - bins: int or 1-D array:
                        if int, auto calculate bin locations, make that number of bins
                        if 1-D array: use these as bin edges
            - bin_logs: T/F: if True, bin log10(variable), else bin variable
            - structure: 3x3 2-D np.ndarray: defines object connectivity
            - wrap:  None, 'sides, 'all: 
                if 'sides', connect structures that span the left/right edge
                if 'all', connect structures that span the left/right edge and top/bottom edge
            - x_sizes, y_sizes: 2-D np.ndarray of shape array.shape: lengths of pixels of array. If None, assume all lengths are 1
        Returns:
            bin_middles, counts: 1-D np.ndarrays of len(bins). If bin_logs, bin_middles will be log10(bin value)
    """
    if x_sizes is None: x_sizes = np.ones(array.shape, dtype=bool)
    if y_sizes is None: y_sizes = np.ones(array.shape, dtype=bool)
    p, a, h, w = get_structure_props(array, x_sizes, y_sizes, structure, wrap=wrap)

    if variable == 'area': to_bin = a
    elif variable == 'perimeter': to_bin = p
    elif variable == 'height': to_bin = h
    elif variable == 'width': to_bin = w
    else: raise ValueError(f'Unsupported variable: {variable}')

    if bin_logs: to_bin = np.log10(to_bin)

    if type(bins) == int: bin_edges = np.linspace(min(to_bin), max(to_bin), bins+1)
    else: bin_edges = bins

    if np.count_nonzero(to_bin>bin_edges[-1])>0: warnings.warn(f'There exist {variable}s outside of bin edges that are being ignored')
    counts, _ = np.histogram(to_bin, bins=bin_edges)

    bin_middles = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    return bin_middles, counts

def edge_effect_dist_exponent(int_counts, edge_counts, log_bin_edges, edge_thresh=.5, min_thresh=10, return_counts=False, min_counts=0):
    """
        Calculate the distribution exponent using linear regression while accounting for effects caused by objects
        truncated by the array edge and effects caused by discretization effects (small clouds).

        Input:
            int_counts: 1-D np array, counts of objects not touching the edge
            edge_counts: 1-D np array, counts of objects touching the edge
            log_bin_edges: 1-D np array, log10(locations) of bins that resulted in the above counts (should be 1 longer than above arrays)
            edge_thresh: omit bins where edge_counts>edge_thresh*total_counts
            min_thresh: omit bins smaller than min_thresh
            min_counts: mark any bin as bad that has counts fewer than min_counts
        Return: 
            exponent, uncertainty
            If return_counts: (exponent, uncertainty), (good_counts, bad_counts)
            Uncertainty cooresponds to 95\%
    """

    log_bin_middles = log_bin_edges[:-1]+0.5*(log_bin_edges[1]-log_bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    good_counts = (int_counts+edge_counts).astype(np.float32) # Do regression on sum of interior and edge

    # Min thresh
    good_counts[10**log_bin_edges[:-1]<min_thresh] = np.nan

    # Edge thresh
    edge_thresh_index = np.argwhere(edge_counts>edge_thresh*good_counts)
    if len(edge_thresh_index) == 0: edge_thresh_index = None
    else: edge_thresh_index = edge_thresh_index[0,0]
    if edge_thresh_index is not None: good_counts[edge_thresh_index:] = np.nan

    # Min count thresh
    good_counts[good_counts<=min_counts] = np.nan

    bad_counts = (int_counts+edge_counts).astype(np.float32)
    bad_counts[np.isfinite(good_counts)] = np.nan
    bad_counts[bad_counts==0] = np.nan  # for putting in log space

    (exponent, _), (error,_) = linear_regression(log_bin_middles, np.log10(good_counts))

    if not return_counts: return (-exponent, error)
    else: return (-exponent, error), (good_counts, bad_counts)

def upscale_props(array, x_sizes, y_sizes, min_pixels=30, edge_behavior='truncate'):
    """
        Given a 2-D array, calculate properties of upscaled matricies. Perimeters include small clouds.

        Input:
            array: single 2-D binary array, may include nans
            x_sizes, y_sizes: 2-D arrays
            min_pixels: limit the upscale factors such that upscaled matrices always have shape >= (min_pixels, min_pixels)
            edge_behavior: 'truncate': do not include perimeter along nans in array
                            'truncate with edges': include perimeter along nans in array
        return: 
            upscale_factors, total_perimeters, cloud_areas, total_numbers, scene_areas
    """

    # # upscale_factors = np.geomspace(1, 1000, 100)  The below generated from this, rounded, duplicates removed, and even numbers replaced with nearby odd numbers. I also added some values above 1001.
    # upscale_factors = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,23,25,27,29,31,33,35,37,41,43,47,51,53,57,61,65,71,75,81,87,93,101,107,115,123,133,143,153,161,175,187,201,215,231,249,267,285,305,327,351,375,405,433,465,499,533,571,615,659,705,757,811,871,933,1001,1073,1149,1233,1321,1417,1519,1629,1747,1873,2009,2155,2311,2477,2657,2849,3053])

    # upscale_factors = np.geomspace(1, 6000, 50)  The below generated from this, rounded, duplicates removed, and even numbers replaced with nearby odd numbers. 
    upscale_factors = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29, 35, 41, 49, 59, 71, 85, 101, 121, 145, 173, 205, 245, 293, 351, 419, 499, 597, 713, 851, 1017, 1213, 1449, 1731, 2067, 2469, 2949, 3523, 4207, 5023, 6001]])

    max_upscale_factor = min(array.shape)/min_pixels
    upscale_factors = upscale_factors[upscale_factors<=max_upscale_factor]

    total_perimeters = []
    cloud_areas = []
    total_numbers = []
    scene_areas = []
    
    for factor in upscale_factors:

        upscaled_array = encase_in_value(upscale_array(array, factor))
        upscaled_x = factor * encase_in_value(upscale_array(x_sizes, factor, make_binary=False))
        upscaled_y = factor * encase_in_value(upscale_array(y_sizes, factor, make_binary=False))

        # total_perimeter() will count only the perimeter along 0s
        if edge_behavior == 'truncate with edges': upscaled_array[np.isnan(upscaled_array)] = 0

        pixel_areas = upscaled_x*upscaled_y
        total_p = total_perimeter(upscaled_array, upscaled_x, upscaled_y)
        total_n = total_number(upscaled_array)
        total_a = np.nansum(pixel_areas[upscaled_array==1])
        scene_a = total_a + np.nansum(pixel_areas[upscaled_array==0])  # cloudy area plus clear area
        total_perimeters.append(total_p)
        cloud_areas.append(total_a)
        total_numbers.append(total_n)
        scene_areas.append(scene_a)
        
        
    return upscale_factors, np.array(total_perimeters), np.array(cloud_areas), np.array(total_numbers), np.array(scene_areas)

def ensemble_upscale_dimension(matricies, min_pixels=30, plot_total_perimeter=False):
    """
        Given a list of 2-D matricies, calculate the ensemble fractal dimension by upscaling. Assume uniform pixel sizes (equal to 1)

        Input:
            matrices: list of 2-D binary matricies
            min_pixels: limit the upscale factors such that upscaled matrices always have shape >= (min_pixels, min_pixels)
            plot_total_perimeter: if True, save plot of total perimeter vs. resolution (in order to check if it is a power law)
        return: 
            D_e, error (95% conf)
    """
    if type(matricies) == np.ndarray: matricies = [matricies]

    # upscale_factors = np.geomspace(1, 1000, 100)  The below generated from this, rounded, duplicates removed, and even numbers replaced with nearby odd numbers
    upscale_factors = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,23,25,27,29,31,33,35,37,41,43,47,51,53,57,61,65,71,75,81,87,93,101,107,115,123,133,143,153,161,175,187,201,215,231,249,267,285,305,327,351,375,405,433,465,499,533,571,615,659,705,757,811,871,933,1001])
    
    max_upscale_factor = min(matricies[0].shape)/min_pixels
    upscale_factors = upscale_factors[upscale_factors<=max_upscale_factor]

    mean_total_perimeters = np.empty((0,upscale_factors.size), dtype=np.float32)
    
    for array in matricies:
        total_perimeters = []
        for factor in upscale_factors:
            upscaled_array = encase_in_value(upscale_array(array, factor), np.nan)
            total_p = total_perimeter(upscaled_array, factor*np.ones_like(array), factor*np.ones_like(array))
            total_perimeters.append(total_p)
        mean_total_perimeters = np.append(mean_total_perimeters, [total_perimeters], axis=0)
    mean_total_perimeters = np.mean(mean_total_perimeters, axis=0)

    (slope, _), (error,_) = linear_regression(np.log10(upscale_factors), np.log10(mean_total_perimeters))
    if plot_total_perimeter:
        import matplotlib.pyplot as plt
        import plotting_functions
        fig, ax = plt.subplots(1,1)
        ax.plot(np.log10(upscale_factors), np.log10(mean_total_perimeters))
        plotting_functions.format_log_log(ax)
        plotting_functions.set_plot_text(title_text='',ytext='Total perimeter',xtext='Resolution')
        plt.tight_layout()
        plotting_functions.savefig('Total perimeter vs resolution')

    return 1-slope, error
    
@njit()
def total_perimeter(array, x_sizes, y_sizes):
    """
        Given a binary array, calculate the total perimeter. Boundary conditions are assumed periodic. 
        Only counts perimeter along edges between 1 and 0. (so that for different b.c. the array could be encased in any other value)
        Assumes periodic B.C.; for something else, padd inputs with 0s or nans.

        Raises ValueError if x_sizes or y_sizes is nan where array is 1
    """
    perimeter = 0
    for (i, j), value in np.ndenumerate(array):
        if value == 1:
            if np.isnan(x_sizes[i,j]) or np.isnan(y_sizes[i,j]): 
                raise ValueError('x_sizes or y_sizes is nan where array is 1')
            if i != array.shape[0]-1 and array[i+1, j] == 0: perimeter += x_sizes[i,j]
            elif i == array.shape[0]-1 and array[0, j] == 0: perimeter += x_sizes[i,j]

            if i != 0 and array[i-1, j] == 0: perimeter += x_sizes[i,j]
            elif i == 0 and array[array.shape[0]-1, j] == 0: perimeter += x_sizes[i,j]

            if j != array.shape[1]-1 and array[i, j+1] == 0: perimeter += y_sizes[i,j]
            elif j == array.shape[1]-1 and array[i, 0] == 0: perimeter += y_sizes[i,j]

            if j != 0 and array[i, j-1] == 0: perimeter += y_sizes[i,j]
            elif j == 0 and array[i, 0] == 0: perimeter += y_sizes[i,j]

    return perimeter

def total_number(array, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
    """
        Given a 2-D array with 0's, nans, and 1's, calculate number of objects of connected 1's where connectivity is defined by structure
    """
    array[np.isnan(array)] = 0
    _, n_structures = label(array.astype(bool), structure, output=np.float32)
    return n_structures

# Cloud resolving model calculations
def MSE(q_v, T, z): 
    """
        Moist Static Energy given
        q_v: Water vapor, g/kg
        T: Temp, K
        z: height, m

        All must be broadcastable to same shape.
    """
    
    return g*z + q_v*L_v + c_p*T



# Base classes
class SaveLoad():
    def __init__(self):
        """
            Parent class for data that needs to be saved and then loaded later. 
            Attributes that are saved must be Numpy arrays.
            Saves any attribute that is an array and is not in self._do_not_save.
            Does not save attributes that start with "_"
            
        """
        # If a variable starts with '_' it will not be saved
        self._save_to_path = dirs.code_files_directory    # directory to save to
        self._filename = None       # unique filename
        self._do_not_save = None    # list of vars to not save. Optional attr
        self._is_complete = None    # bool: will not save unless True


    def load_data(self):
        """
            Output:
                True if files found and loaded, False otherwise

            Load all attribute variables associated with current instance in folder at path,
            or load previous save if files exist. Will load variables even 
            if they are not defined in __init__.

        """
        if self._filename[-1] != '/': self._filename += '/'
        if os.path.exists(self._save_to_path+self._filename): 
            for attr_name in os.listdir(self._save_to_path+self._filename):
                try:
                    attr = np.load(self._save_to_path+self._filename+attr_name)
                    attr_name = attr_name[:-4]  # remove .npy
                except Exception as e:
                    print(e)
                    print('Failed to open file ',self._save_to_path+self._filename+attr_name)
                    exit()
                setattr(self, attr_name, attr)
            return True
        else: return False

    def save_data(self):
        """
            Input:
                save_everything - T/F Whether to also save data arrays.
            Output:
                True if files have been overwritten, False otherwise

            Save all attribute variables associated with current instance in folder at path.
            Will save variables even  if they are not defined in __init__.
            Will overwrite data.
            Raises ValueError if calculations have not been run.
            Does not save variables that stat with '_'
        """
        if self._filename[-1] != '/': self._filename += '/'
        
        if os.path.exists(self._save_to_path+self._filename): to_return = True
        else: 
            os.mkdir(self._save_to_path+self._filename)
            to_return = False
        if not self._is_complete: raise ValueError('All calculations must be run before saving')
        for attr_name in vars(self):
            if (not attr_name.startswith("_") and (type(getattr(self, attr_name)) == np.ndarray)):
                if "_do_not_save" in vars(self) and (attr_name in self._do_not_save): continue     
                attr = getattr(self, attr_name)
                if attr_name in os.listdir(self._save_to_path+self._filename): os.remove(self._save_to_path+self._filename+attr_name)   # if already there, np would have appended instead of overwritten
                np.save(self._save_to_path+self._filename+attr_name, attr, allow_pickle=False)
        
        return to_return
    