"""
    Functions for calculating size distributions in 2-D domains while taking into account finite size effects.
    By Thomas DeWitt (https://github.com/thomasdewitt/)
"""
import numpy as np
from scipy.ndimage import label
from numba import njit, List
from warnings import warn
from skimage.segmentation import clear_border


# Functions for calculating size distributions

def finite_array_size_distribution(arrays, variable, x_sizes=None, y_sizes=None, bins=100, bin_logs=True, min_threshold=10, truncation_threshold=0.5):
    """
        Calculate the size distributions for structures within a 
        list of binary arrays, where 'size' is perimeter, area, length, or width.
        Returns the size distributions for truncated objects and nontruncated objects
        and the index where truncated object begin to dominate.
        
        Works for binary arrays and also for binary arrays where the data boundary is 
        demarcated by nans. This enables the domain boundary to be an arbitrary shape, 
        rather than be rectangular (as is the case for a binary array).
        
        Input:
            - arrays: 2-D np.ndarray or list of 2-D np.ndarray, where objects of interest have value 1, 
                        the background has value 0, and no data has np.nan. 
                        Interior nans are treated like 0's, except the perimeter along them is not counted.
            - variable: 'area','perimeter','height','width': which object attribute to bin by. See below for definitions.
            - x_sizes, y_sizes: 2-D np.ndarray of shape array.shape: lengths of pixels of array. If None, assume all lengths are 1
            - bins: int or 1-D array:
                        if int, auto calculate bin locations, make that number of bins
                        if 1-D array: use these as bin edges or log10(bin edges). They must be uniformly 
                        linearly or logarithmically spaced (depending on bin_logs)
            - bin_logs: T/F: if True, bin log10(variable) into logarithmically-spaced bins. If False, bin
                        variable into linearly spaced bins (if bins are explicitely passed, use these in any case)
            - min_threshold: smallest bin edge. If bin edges are passed, this arg is ignored.
            - truncation_threshold: float between 0 and 1. Bins with a larger fraction of truncated objects than this are omitted from the regression
        Output:
            - bin_middles, nontruncated_counts, truncated_counts, truncation_index
                Note: if bin_logs is True, bin middles is actually log10(bin_middles)

        Notes:

        'variable' definitions: 
            'perimeter': Sum of pixel edge lengths between all pixels within a structure and 
                        neighboring values of 0. Does not include perimeter adjacent to a nan.
                        A donut shaped structure returns a single value.
            'area': Sum of individual pixel areas constituting the structure
            'length' or 'width': Overall distance between the farthest two points in a structure in
                                the x- or y- direction.

    """
    if type(arrays) == np.ndarray: arrays = [arrays]
    if x_sizes is None: x_sizes = np.ones(arrays[0].shape, dtype=bool)
    if y_sizes is None: y_sizes = np.ones(arrays[0].shape, dtype=bool)


    if type(bins) == int: 
        max_value = np.nansum(x_sizes*y_sizes)
        if bin_logs: bin_edges = np.linspace(np.log10(min_threshold), np.log10(max_value), bins+1)
        else: bin_edges = np.linspace(min_threshold, max_value, bins+1)
    else: bin_edges = bins

    truncated_counts = np.zeros(bin_edges.size-1)
    nontruncated_counts = np.zeros(bin_edges.size-1)

    for array in arrays:
        # Encase the array in nans to ensure objects in contact with the edge are considered truncated
        array = encase_in_value(array)

        no_truncated = remove_structures_touching_border_nan(array)
        truncated_only = array-no_truncated

        truncated_counts += array_size_distribution(truncated_only, x_sizes=encase_in_value(x_sizes), y_sizes=encase_in_value(y_sizes), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]
        nontruncated_counts += array_size_distribution(no_truncated, x_sizes=encase_in_value(x_sizes), y_sizes=encase_in_value(y_sizes), variable=variable, wrap=None, bins=bin_edges, bin_logs=bin_logs)[1]

    # Find index where number of edge clouds is greater than threshold times total number of clouds
    truncation_index = np.argwhere(truncated_counts>truncation_threshold*(truncated_counts+nontruncated_counts))
    if truncation_index.size == 0:     # then there is no need to truncate
        truncation_index = len(bin_edges)
    else: truncation_index = truncation_index[0,0]

    bin_middles = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    return bin_middles, nontruncated_counts, truncated_counts, truncation_index

def finite_array_powerlaw_exponent(arrays, variable, x_sizes=None, y_sizes=None, bins=100, min_threshold=10, truncation_threshold=0.5, return_counts=False):
    """
        Calculate the power-law exponent for size distributions of structures within a 
        list of binary arrays, where 'size' phi can be perimeter, area, length, or width:

        n(phi) \propto phi^{-(1+exponent)}

        
        Works for binary arrays and also for binary arrays where the data boundary is 
        demarcated by nans. This enables the domain boundary to be an arbitrary shape, 
        rather than be rectangular (as is the case for a binary array).
        
        Input:
            - arrays: 2-D np.ndarray or list of 2-D np.ndarray, where objects of interest have value 1, 
                        the background has value 0, and no data has np.nan. 
                        Interior nans are treated like 0's, except the perimeter along them is not counted.
            - variable: 'area','perimeter','height','width': which object attribute to bin by. See below for definitions.
            - x_sizes, y_sizes: 2-D np.ndarray of shape array.shape: lengths of pixels of array. If None, assume all lengths are 1
            - bins: int or 1-D array:
                        if int, auto calculate bin locations, make that number of bins
                        if 1-D array: use these as log10(bin edges). They must be uniformly logarithmically spaced.
            - min_threshold: smallest bin edge. If bin edges are passed, this arg is ignored.
            - truncation_threshold: float between 0 and 1. Bins with a larger fraction of truncated objects than this are omitted from the regression
        Output:
            if return_counts:
                return (exponent, error), (log10(bin_middles), log10(good_counts))
                    where good_counts are the total counts for values smaller than the truncation threshold
                    error corresponding to 95% conf. interval
            else:
                return (exponent, error):
                    error corresponding to 95% conf. interval

        Notes:

        'variable' definitions: 
            'perimeter': Sum of pixel edge lengths between all pixels within a structure and 
                        neighboring values of 0. Does not include perimeter adjacent to a nan.
                        A donut shaped structure returns a single value.
            'area': Sum of individual pixel areas constituting the structure
            'length' or 'width': Overall distance between the farthest two points in a structure in
                                the x- or y- direction.
    """

    log_bin_middles, nontruncated_counts, truncated_counts, truncation_index = finite_array_size_distribution(arrays=arrays, 
                                                                                             variable=variable, 
                                                                                             x_sizes=x_sizes, 
                                                                                             y_sizes=y_sizes, 
                                                                                             bins=bins, 
                                                                                             bin_logs=True, 
                                                                                             min_threshold=min_threshold, 
                                                                                             truncation_threshold=truncation_threshold)
    
    if log_bin_middles[truncation_index]-np.log10(min_threshold)<2:
        warn(f'Power law exponent is being estimated using data spanning only {log_bin_middles[truncation_index]-np.log10(min_threshold):.01f} orders of magnitude')

    total_good_counts = (truncated_counts+nontruncated_counts)[:truncation_index]

    log_bin_middles = log_bin_middles[:truncation_index]

    total_good_counts[total_good_counts==0] = np.nan    # eliminate log of 0 warning

    log_total_good_counts = np.log10(total_good_counts)
    
    (slope, _), (slope_error, _) = linear_regression(log_bin_middles, log_total_good_counts)

    if return_counts:
        return (-slope, slope_error), (log_bin_middles, log_total_good_counts)
    return -slope, slope_error

def array_size_distribution(array, variable='area', bins=30, bin_logs=True, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), wrap=None, x_sizes=None, y_sizes=None):
    """
        Given a single binary array, calculate contiguous object sizes and bin them by area/perimeter/length/width

        Note: this function does not account for object truncation by the domain boundary.

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
        Output:
            - bin_middles, counts: 1-D np.ndarrays of len(bins). If bin_logs, bin_middles will be log10(bin value)
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

    if np.count_nonzero(to_bin>bin_edges[-1])>0: warn(f'There exist {variable}s outside of bin edges that are being ignored')
    counts, _ = np.histogram(to_bin, bins=bin_edges)

    bin_middles = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    return bin_middles, counts

# Helper functions

def label_periodic_boundaries(labelled_array, wrap):
    """
        This functions makes labelled structures that span the edge have the same label.

        Parameters:
        labelled_array (numpy.ndarray): A 2D array where each unique non-zero element represents a distinct label. Should be the output of scipy.ndimage.label().
        wrap (str): A string that determines how the boundaries of the array should be wrapped. 
                    It can take three values: 'sides', 'both', or any other string.

        If 'wrap' is 'sides' or 'both':
            The function sets the labels on the right boundary to be the same as those on the left boundary.

        If 'wrap' is 'both':
            The function also sets the labels on the top boundary to be the same as those on the bottom boundary.

        If 'wrap' is neither 'sides' nor 'both':
            The function raises a ValueError.

        Returns:
        labelled_array (numpy.ndarray): The input array with its periodic boundaries labelled as per the 'wrap' parameter.

        Raises:
        ValueError: If 'wrap' is neither 'sides' nor 'both'.
    """
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
    if wrap not in ['sides','both']: raise ValueError(f'wrap = {wrap} not supported')
    return labelled_array

def get_structure_props(array, x_sizes, y_sizes, structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), print_none=False, wrap=None):
    """
        Input:
            array - Binary array of strc: 2-d np.ndarray, padded with 0's or np.nan's
            x_sizes = Sizes of pixels in horizontal direction, same shape as array: 2-d np.ndarray
            y_sizes = Sizes of pixels in vertical direction, same shape as array: 2-d np.ndarray
            structure = Defines connectivity
            print_none = Print message if no structures found
            wrap = None, 'sides', 'both': 
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


def linear_regression(x, y):
    """
        Return (slope, y-int), (error_slope, error_y_int) for 95% conf
    """
    if type(x) != np.ndarray or type(y) != np.ndarray: raise TypeError('x, y, must be of type np.ndarray')
    index = np.isfinite(x) & np.isfinite(y)
    if len(x[index]) <3:    # "the number of data points must exceed order to scale the covariance matrix"
        warn('Less than 3 points (x,y) are good (not nan), returning nans')
        return (np.nan, np.nan),(np.nan, np.nan)
    try:
        coefficients, cov = np.polyfit(x[index], y[index], 1, cov=True)
        error = np.sqrt(np.diag(cov))
    except Exception as e:
        warn('Linear regression failed, error message\n','     ',e)
        return (np.nan, np.nan),(np.nan, np.nan)
    return coefficients, 2*error  # 95% conf interval is 2 times standard error 

def remove_structures_touching_border_nan(array):
    """
        Input:
            array: 2-D np.ndarray consisting of 0s, 1s, and np.nan. All values at the array edge should be np.nan
        Output:
            2-D np.ndarray consisting of 0s, 1s, and np.nan with any structure in contact with the nan 
            values around the outer edge of the good data removed
            "in contact" is defined using adjacent connectivity, i.e. 4-connectivity
        
    """
    if array.ndim != 2: raise ValueError('array not 2-dimensional')

    nanmask = np.isnan(array).astype(int)
    edge_nan_mask  = (nanmask - clear_border_adjacent(nanmask)).astype(bool)

    with_edge = array.copy()
    with_edge[edge_nan_mask] = 1

    cleared = clear_border_adjacent(with_edge).astype(float)
    cleared[edge_nan_mask] = np.nan
    cleared[np.isnan(array)] = np.nan
    return cleared
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
def remove_structure_holes(array, periodic=False):
    """
        Fills in all holes in all structures within array.

        Set any value of 0 that is not connected to the largest connected structure of 0s (the background) to 1.

        Assume the largest contiguous area of 0s is the "background".


        Input:
            array: 2D np.ndarray with values either 0,1, or np.nan
            periodic: False, 'both', 'sides':
                For structures lying along the boundary, if periodic=False, the behavior is as if the array was padded with 1's, i.e. holes that are connected to the edge are filled.

        Output: filled array
    """
    if type(array) != np.ndarray: raise ValueError('array must be a np.ndarray object')
    filled = array.copy()
    filled[np.isnan(filled)] = 0
    if np.any(filled>1): raise ValueError('array can only have values 0, 1, or np.nan')
    
    # invert and label
    labelled, _ = label((1-filled))
    if periodic != False: labelled = label_periodic_boundaries(labelled, periodic)
    # largest structure will be the background or the cloudy areas.
    unique_values, unique_counts = np.unique(labelled.flatten(), return_counts=True)
    # Make sure we don't identify the cloudy areas as the background.
    unique_counts, unique_values = unique_counts[unique_values!=0], unique_values[unique_values!=0]
    label_of_background = unique_values[unique_counts.argmax()]

    filled[(labelled != 0) & (labelled != label_of_background)] = 1

    if np.count_nonzero(np.isnan(array))>0: filled[np.isnan(array)] = np.nan

    return filled
     
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
