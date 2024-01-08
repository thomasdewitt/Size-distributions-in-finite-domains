"""
    Author: Thomas DeWitt
    Functions relating to the edge problem paper, determining
"""
from PyThomas import useful_functions as uf
import numpy as np

def subarray_area_distributions(arrays, split_divisions_x, split_divisions_y, periodic_boundaries=False, pixel_side_lengths=1, bin_edges = np.linspace(1,6, 51), bin_logs=True):
    """
    Calculate area distributions of truncated and non-truncated objects within arrays.

    This function computes area distributions of truncated and non-truncated objects for 
    sub-arrays, which are split from the parent array by .
    The distributions are calculated in terms of provided bin edges.

    Args:
        arrays (list of numpy.ndarray): List of input arrays, which are each 2-D and binary to analyze.
        split_divisions_x (list of int): List of divisions along the x-axis (columns) to split array into sub-arrays.
        split_divisions_y (list of int): List of divisions along the y-axis (rows) to split array into sub-arrays.
        periodic_boundaries (bool, optional): Whether to connect objects periodically along the boundary.
                                             Defaults to False.
        pixel_side_lengths (float, optional): Length of a pixel's side. Assumed constant. Defaults to 1.
        bin_edges (numpy.ndarray, optional): Bin edges for calculating area distributions.
                                             Defaults to np.linspace(1, 6, 51).
        bin_logs (bool, optional): If True, take log10 of area values before binning. Passed to uf.array_size_distribution(). Default is True.

    Returns:
        tuple: A tuple containing two lists (interior and edge), each element corresponding to the 
                counts of structures in the corresponding subarrays


    Example:
        arrays = [array([[1, 0, 1, 0],
                         [0, 1, 0, 1]]),
                  array([[1, 1],
                         [0, 0]])]
        split_divisions_x = [2, 1]
        split_divisions_y = [2, 1]
        interior, edge = subarray_area_distributions(arrays, split_divisions_x, split_divisions_y)
        # interior contains a list of interior area distribution arrays for each division pair
        # edge contains a list of edge area distribution arrays for each division pair
    """
    if not all(isinstance(arr, np.ndarray) for arr in arrays) or not isinstance(arrays, list):
        raise ValueError("Input 'arrays' must be a list of numpy arrays.")
    
    all_interior_area_counts = []
    all_edge_area_counts = []
    if periodic_boundaries: wrap = 'both'
    else: wrap = None

    for div_x, div_y in zip(split_divisions_x, split_divisions_y):
        interior_area_counts = np.zeros(bin_edges.size-1)
        edge_area_counts = np.zeros(bin_edges.size-1)
        for array in arrays:
            for subarray in uf.split_array(array, div_x, div_y):
                if wrap is None:
                    int_only = uf.clear_border_adjacent(subarray)
                    edge_only = subarray-int_only
                    interior_area_counts += uf.array_size_distribution(int_only, bins=bin_edges, bin_logs=bin_logs, wrap=wrap, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths)[1]
                    edge_area_counts += uf.array_size_distribution(edge_only, bins=bin_edges, bin_logs=bin_logs, wrap=wrap, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths)[1]
                else: 
                    interior_area_counts += uf.array_size_distribution(subarray, bins=bin_edges, bin_logs=bin_logs, wrap=wrap, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths)[1]
        all_interior_area_counts.append(interior_area_counts)
        all_edge_area_counts.append(edge_area_counts)

    return all_interior_area_counts, all_edge_area_counts

def remove_truncation_affected_bins(interior_clouds, edge_clouds, edge_thresh=0.5, bin_edges = None, return_both=True):
    """
        This function removes clouds that are larger than a specified edge threshold. The edge threshold is defined as the bin in which 
        the number of edge clouds is greater than the edge threshold multiplied by the sum of the total counts. The function 
        uses logarithmically spaced bins for this operation.

        Parameters:
        interior_clouds (np.array): An array containing the interior clouds data.
        edge_clouds (np.array): An array containing the edge clouds data.
        edge_thresh (float, optional): The edge threshold value. Default is 0.5.
        bin_edges (np.array, optional): The edges of the bins to be used. If None, the function will compute it. Default is None.
        return_both (bool, optional): If True, the function returns both interior and edge clouds that are less than the maximum value. 
                                    If False, it returns only the interior clouds that are less than the maximum value. Default is True.

        Returns:
        np.array: An array of clouds that are less than the maximum value.
    """
    interior_clouds = np.array(interior_clouds)
    edge_clouds = np.array(edge_clouds)

    max_value = find_edge_thresh(interior_clouds, edge_clouds, edge_thresh=edge_thresh, bin_edges = bin_edges) 

    if return_both: return np.append(interior_clouds[interior_clouds<max_value], edge_clouds[edge_clouds<max_value])
    else: return interior_clouds[interior_clouds<max_value]

def find_edge_thresh_values(interior_clouds, edge_clouds, edge_thresh=0.5, bin_edges = None):
    """
        This function finds the edge threshold for a given set of interior and edge clouds. The edge threshold is defined as the bin in which 
        the number of edge clouds is greater than the edge threshold multiplied by the total counts. The function 
        uses logarithmically spaced bins for this operation.

        Parameters:
        interior_clouds (np.array): An array containing the interior clouds data.
        edge_clouds (np.array): An array containing the edge clouds data.
        edge_thresh (float, optional): The edge threshold value. Default is 0.5.
        bin_edges (np.array, optional): The edges of the bins to be used. If None, the function will compute it. Default is None.

        Returns:
        float: The computed edge threshold.
        
        Raises:
        ValueError: If either interior_clouds or edge_clouds contain NaN or zero values.
    """

    log_interior_clouds = np.log10(interior_clouds)
    log_edge_clouds = np.log10(edge_clouds)

    if np.count_nonzero(~np.isfinite(log_interior_clouds)) + np.count_nonzero( ~np.isfinite(log_edge_clouds))>0: raise ValueError('interior_clouds or edge_clouds contain nan or 0')

    if bin_edges is None:
        bin_edges = np.linspace(min(log_interior_clouds.min(), log_edge_clouds.min()), max(log_interior_clouds.max(), log_edge_clouds.max()), num=51)


    # Compute histogram counts for interior and edge clouds
    interior_counts = np.histogram(log_interior_clouds, bins=bin_edges)[0]
    edge_counts = np.histogram(log_edge_clouds, bins=bin_edges)[0]

    return find_edge_thresh_counts(interior_counts, edge_counts, bin_edges, edge_thresh)

def find_edge_thresh_counts(interior_counts, edge_counts, bin_edges, edge_thresh=0.5):
    """
        This function finds the edge threshold for a given set of interior and edge clouds when ALREADY binned. The edge threshold is defined as the bin in which 
        the number of edge clouds is greater than the edge threshold multiplied by the total counts. The function 
        uses logarithmically spaced bins for this operation.

        Input:
            bin_edges should be the log of the actual bin edges
            The counts should be calculated as

                interior_counts = np.histogram(np.log10(interior_clouds), bins=bin_edges)[0]
                edge_counts = np.histogram(np.log10(edge_clouds), bins=bin_edges)[0]
            
                if interior_clouds and edge_clouds contain the values of the cloud sizes.
        Returns:
        float: The computed edge threshold.
        
    """

    # Find index where number of edge clouds is greater than threshold times total number of clouds
    index = np.argwhere(edge_counts>edge_thresh*(edge_counts+interior_counts))
    
    if index.size == 0:     # then there is no need to truncate
        return np.inf
    else:
        return 10**bin_edges[index[0,0]]  # this will be left bin edge. Exponentiate because we took log


def subarray_area_values(arrays, split_divisions_x, split_divisions_y, periodic_boundaries=False, pixel_side_lengths=1):

    if not all(isinstance(arr, np.ndarray) for arr in arrays) or not isinstance(arrays, list):
        raise ValueError("Input 'arrays' must be a list of numpy arrays.")
    
    all_interior_areas = []
    all_edge_areas = []
    if periodic_boundaries: wrap = 'both'
    else: wrap = None

    for div_x, div_y in zip(split_divisions_x, split_divisions_y):
        interior_areas = []
        edge_areas = []
        for array in arrays:
            for subarray in uf.split_array(array, div_x, div_y):
                if wrap is None:
                    int_only = uf.clear_border_adjacent(subarray)
                    edge_only = subarray-int_only
                    interior_areas.extend(uf.get_structure_props(int_only, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths)[1])
                    edge_areas.extend(uf.get_structure_props(edge_only, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths)[1])
                else: 
                    interior_areas.extend(uf.get_structure_props(subarray, x_sizes=np.ones_like(subarray)*pixel_side_lengths, y_sizes=np.ones_like(subarray)*pixel_side_lengths, wrap='both')[1])
        all_interior_areas.append(interior_areas)
        all_edge_areas.append(edge_areas)

    return all_interior_areas, all_edge_areas