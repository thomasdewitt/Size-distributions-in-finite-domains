# Size-distributions-in-finite-domains
### Software to calculate size distributions for objects in 2-D domains using methodology described in DeWitt &amp; Garrett (2024): <u>Finite domains cause bias in measured and modeled distributions of cloud sizes</u>.

Link to the paper [here](https://acp.copernicus.org/articles/24/8457/2024/acp-24-8457-2024.html).

The software available here are general purpose Python scripts that implement the methodology recommended by DeWitt &amp; Garrett (2024). For the specific code used to generate figures in that publication (for example, code that processes satellite imagery), please contact Thomas DeWitt directly.

There are two main functions that can be used to calculate size distributions in `size_distributions.py`, described below.

The first calculates the power law exponent for 2-D arrays containing structures, which are groupings of array values of 1, while reducing bias caused by the finite domain size. The second calculates the size distributions of truncated and non-truncated structures from which parameters for more general size distributions can be calculated.

`size_distributions.finite_array_powerlaw_exponent(arrays, variable, x_sizes=None, y_sizes=None, bins=100, min_threshold=10, truncation_threshold=0.5, return_counts=False)`

  Calculate the power-law exponent $\alpha$ for size distributions of structures within a 
  list of binary arrays, where 'size' $\phi$ can be perimeter, area, length, or width:

  $n(\phi) \propto \phi^{-(1+\alpha)}$
  
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
                                
`finite_array_size_distribution(arrays, variable, x_sizes=None, y_sizes=None, bins=100, bin_logs=True, min_threshold=10, truncation_threshold=0.5)`

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




