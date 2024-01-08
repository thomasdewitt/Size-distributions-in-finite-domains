# Size-distributions-in-finite-domains
### Software that produced the results of DeWitt &amp; Garrett (2024) (Finite domains cause bias in measured and modeled distributions of cloud sizes).

Link to the preprint: (coming soon)

The Python files `linear_regression_testing.py` and `plot_edge_effect.py` produce the figures and tables published in DeWitt &amp; Garrett (2024).

The file `edge_effect_functions.py` contains some functions that should be useful out-of-the-box for those wanting to implement our recommendations in their project. For example, functions included can identify where the 50% threshold for bins that are contaminated by truncated objects are.

The file `useful_functions.py` contains some custom functions that are needed to calculate size distributions in 2-D domains. 

The file `ising_model.py` was used to simulate percolation lattices. It can also simulate the Ising model somewhat efficiently, though this functionality was not used in DeWitt &amp; Garrett (2024).

The other files, `n2str.py`, `tex_funcs.py`, and `plotting_functions.py`, contain useful functions to make nicely formatted numbers, LaTeX tables, and MatPlotLib plots, respectively.


If you would like to run these scripts on your machine, several things are needed. First, make sure the higher level scripts `linear_regression_testing.py`, `ising_model.py`, and `plot_edge_effect.py` import the others listed here correctly. Also, you will have to hard-code directories into the scripts for functions that read/write files. Finally, make sure to download GOES data from ICARE (https://www.icare.univ-lille.fr/) if you would like to use cloud data.
