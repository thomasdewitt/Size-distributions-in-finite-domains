import numpy as np
import edge_effect_functions as ef
from PyThomas import useful_functions as uf
import data_reader
import plotting_functions as pf
import matplotlib.pyplot as plt
import ising_model
import n2str
import time
import tex_funcs
from scipy.optimize import curve_fit

single_fig_size = (4.5,2.5)     # one rectangular plot
double_fig_size = (6.5,3)
font_small = 8
font_large = 10
capsize = 5
scatter_size = 7
lw = .8
color1 = "#FF4D00"
color2 = "#E8005E"
color3 = "#701EFF"
color4 = "#166AEB"
color5 = "#12F3EA"
# colors = ['#332287','#117833','#CC6677','#AB449A']
colors = ['#CD0089','#5ADB87','#FFAE00','#00B4FF']
# colors = [color1,color3,color4,color5]
markers = ['s','d','^','o']

# Same for all plots
nbins = 50
bin_edges = np.linspace(1,6, nbins+1)

bootstrap_MLE = True

def main():

    start = time.time()

    example_bad_alpha_calc(10)      # Table 1 and Fig. 7
    edge_effect_alphas('GOES',10)   # Table 2
    edge_effect_alphas('Percolation',1)    # Table 2

    # edge_effect_plot('GOES', nlattices=10)    # Fig 4
    # edge_effect_plot('Percolation', nlattices=3)   # Fig 5
    # edge_v_int_plot(10)                     # Fig. 6
    # periodic_edge_effect_plot(3)        # Fig. 8
    # exponential_percolation_plot(3, log_bin=True)   # Fig. 9
    # exponential_percolation_plot(3, log_bin=False)      # Fig. C1

    print('Total time: {:.0f}s ({:.1f}m or {:.1f}h or {:.1f}d)'.format(time.time()-start,(time.time()-start)/60, (time.time()-start)/3600,(time.time()-start)/(24*3600)))

class DataContainer(uf.SaveLoad):
    def __init__(self, model, n_lattices = 10, periodic_boundaries=False, div_to_save=None):
        self.model = model
        self.n_lattices = n_lattices
        self.periodic = periodic_boundaries
        self.min_thresh = 10
        self.div_to_save = div_to_save  # index of self.split_divisions that we want to save areas for


        if model == 'GOES':
            self.split_divisions = (1,10,40,100)
            self.resolution = 2
            # domain sizes: 2000 by 2000 pixels, at res of 2km
            self.domain_sizes_label = [f'{4000//i}$\\times${4000//i} km domain' for i in self.split_divisions]
        elif model == 'Percolation':
            self.split_divisions = (1,25,100,250)
            self.resolution = 1
            self.domain_sizes_label = [f'{10000//i}$\\times${10000//i} pixel domain' for i in self.split_divisions]

        
        self.bin_edges = bin_edges
        self.bin_middles = self.bin_edges[:-1]+(self.bin_edges[1]-self.bin_edges[0])/2
        self.interior_counts = np.array([np.zeros(nbins) for _ in self.split_divisions])
        self.edge_counts = np.array([np.zeros(nbins) for _ in self.split_divisions])

        # Values of cloud areas. Only saved for self.split_divisions[div_to_save]
        self.interior_areas = []
        self.edge_areas = []

        self._filename = f'EdgeEffectDataContainer_{model}_{n_lattices}_periodic={periodic_boundaries}_divtosave={div_to_save}'
        self._save_to_path = pf.dirs.code_files_directory

        # if True:
        if not self.load_data():
            self.get_data()
            self.calculate_distributions()
            self._is_complete = True
            self.save_data()

    def get_data(self):

        if self.model == 'GOES':
            dates = [str(2021152+i) for i in range(self.n_lattices)]
            self.scenes = data_reader.GEOST_filenames(dates, satellite='GOES WEST',nearest_time='day')
        elif self.model == 'Percolation':
            self.scenes = [ising_model.Ising((10000,10000), iterations_per_gridpoint=0, identifier=i) for i in range(self.n_lattices)]

    def calculate_distributions(self):
        dates = []
        for i,scene in enumerate(self.scenes):
            print(f'Calculating {self.model} distributions {i+1}/{len(self.scenes)}')
            
            if self.model == 'GOES':
                scene.load_files()
                # use only middle 2000x2000 pixels
                good_ind = [5426//2-1000,5426//2+1000]
                cloud_mask = scene.cloud_mask[good_ind[0]:good_ind[-1],good_ind[0]:good_ind[-1]]
                lat = scene.lat[good_ind[0]:good_ind[-1],good_ind[0]:good_ind[-1]]
                lon = scene.lon[good_ind[0]:good_ind[-1],good_ind[0]:good_ind[-1]]
                maxlat, minlat = np.nanmax(lat), np.nanmin(lat)
                maxlon, minlon = np.nanmax(lon), np.nanmin(lon)
                dates.append(scene.date)
                # make sure there are no nans
                if np.count_nonzero(np.isnan(cloud_mask))>0:
                    raise ValueError(f'Nans found in cloud mask for date {scene.date}')
                cloud_mask = cloud_mask.astype(int)
            elif self.model == 'Percolation':
                scene.load_data()
                cloud_mask = scene.lattice.astype(np.int16)

            # calculate counts
            new_int_counts, new_edge_counts = ef.subarray_area_distributions([cloud_mask], self.split_divisions, self.split_divisions, self.periodic, self.resolution, self.bin_edges)

            # calculate area values
            new_int_areas, new_edge_areas = ef.subarray_area_values([cloud_mask], self.split_divisions, self.split_divisions, False, self.resolution)
            if self.div_to_save is not None:
                self.interior_areas.extend(new_int_areas[self.div_to_save])
                self.edge_areas.extend(new_edge_areas[self.div_to_save])

            # add counts to data structure
            for j in range(len(self.split_divisions)):
                self.interior_counts[j] += new_int_counts[j]
                self.edge_counts[j] += new_edge_counts[j]

            self.scenes[i] = None
        if self.model == 'GOES':
            print(f'Lat: {minlat:.02f}, {maxlat:.02f}')
            print(f'Lon: {minlon:.02f}, {maxlon:.02f}')
            for date in dates: print(f'         {date}')


        # remove counts less than minimum threshold
        for i in range(len(self.split_divisions)):
            self.interior_counts[i][(10**self.bin_edges[1:]<self.min_thresh*self.resolution**2)] = np.nan
            self.edge_counts[i][(10**self.bin_edges[1:]<self.min_thresh*self.resolution**2)] = np.nan

        self.interior_areas = np.array(self.interior_areas)
        self.edge_areas = np.array(self.edge_areas)

class ExpDataContainer(uf.SaveLoad):
    def __init__(self, n_lattices = 10, periodic_boundaries=False, div_to_save=None, log_bin=False):
        self.n_lattices = n_lattices
        self.periodic = periodic_boundaries
        self.min_thresh = 10
        self.div_to_save = div_to_save  # index of self.split_divisions that we want to save areas for
        self.lattice_size = 10000
        self.log_bin = log_bin
        self.occupation_probability = 0.5

        self.split_divisions = (1,40,250)
        self.resolution = 1
        self.domain_sizes_label = [f'{self.lattice_size//i}$\\times${self.lattice_size//i} pixel domain' for i in self.split_divisions]

        if self.log_bin:
            self.bin_edges = np.linspace(1, 4.5, nbins+1)
        else:
            self.bin_edges = np.linspace(0,800, nbins+1)
        self.bin_middles = self.bin_edges[:-1]+(self.bin_edges[1]-self.bin_edges[0])/2
        self.interior_counts = np.array([np.zeros(nbins) for _ in self.split_divisions])
        self.edge_counts = np.array([np.zeros(nbins) for _ in self.split_divisions])

        self._filename = f'EdgeEffectDataContainer_Percolation_Exponential_{n_lattices}_periodic={periodic_boundaries}_divtosave={div_to_save}_size={self.lattice_size}_log_bin={log_bin}_prob={self.occupation_probability}'
        self._save_to_path = pf.dirs.code_files_directory

        # if True:
        if not self.load_data():
            self.get_data()
            self.calculate_distributions()
            self._is_complete = True
            self.save_data()

    def get_data(self): self.scenes = [ising_model.Ising((self.lattice_size,self.lattice_size), starting_probability=self.occupation_probability, iterations_per_gridpoint=0, identifier=i) for i in range(self.n_lattices)]

    def calculate_distributions(self):

        for i,scene in enumerate(self.scenes):
            print(f'Calculating exponential percolation distributions {i+1}/{len(self.scenes)}')
            if not scene._is_complete: 
                scene.create_lattice()
                scene.save_data()
            
            scene.load_data()
            cloud_mask = scene.lattice.astype(np.int16)

            # calculate counts
            new_int_counts, new_edge_counts = ef.subarray_area_distributions([cloud_mask], self.split_divisions, self.split_divisions, self.periodic, self.resolution, self.bin_edges, bin_logs=self.log_bin)

            # add counts to data structure
            for j in range(len(self.split_divisions)):
                self.interior_counts[j] += new_int_counts[j]
                self.edge_counts[j] += new_edge_counts[j]


            self.scenes[i] = None
        # remove counts less than minimum threshold
        for i in range(len(self.split_divisions)):
            self.interior_counts[i][(10**self.bin_edges[1:]<self.min_thresh*self.resolution**2)] = np.nan
            self.edge_counts[i][(10**self.bin_edges[1:]<self.min_thresh*self.resolution**2)] = np.nan


def example_bad_alpha_calc(nlattices=10):

    subarray_id = 2
    range_to_calculate_alpha_subdomain = (20,800)


    error_label = 'Difference'
    # error_label = '\\begin{tabular}[c]{@{}l@{}} Difference \\\\ LR $\\hat{\\alpha}$\end{tabular}'
    
    table_header = ['Domain Size','Fit Range','\\begin{tabular}[c]{@{}l@{}} Excluding Truncated \\\\ LR $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Including Truncated \\\\ LR $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Excluding Truncated \\\\ MLE $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Including Truncated \\\\ MLE $\\hat{\\alpha}$\end{tabular}']
    table_data = []
    
    # First, calculate alpha for the subdomains, then the original domain

    alphahats = []
    alphahaterrors = []

    for id in [0, subarray_id]:

        data = DataContainer('GOES', nlattices, False, div_to_save=id)
        data.interior_areas = np.array(data.interior_areas)
        data.edge_areas = np.array(data.edge_areas)

        if id == 0: range_to_calculate_alpha = (20,ef.find_edge_thresh_values(data.interior_areas, data.edge_areas, edge_thresh=0.5))
        # if id == 0: range_to_calculate_alpha = range_to_calculate_alpha_subdomain
        else: range_to_calculate_alpha = range_to_calculate_alpha_subdomain

        interior_areas = data.interior_areas[(data.interior_areas>range_to_calculate_alpha[0]) & (data.interior_areas<range_to_calculate_alpha[1])] 
        edge_areas = data.edge_areas[(data.edge_areas>range_to_calculate_alpha[0]) & (data.edge_areas<range_to_calculate_alpha[1])] 

        all_areas = np.append(edge_areas, interior_areas)

        if bootstrap_MLE:
            alpha_estimate_MLE_interior = uf.bootstrap(interior_areas, uf.estimate_truncated_power_law_exponent_MLE, 200)
            alpha_estimate_MLE_all = uf.bootstrap(all_areas, uf.estimate_truncated_power_law_exponent_MLE, 200)
        else:
            print('Incorrect uncertainty for MLE (did not bootstrap)')
            alpha_estimate_MLE_interior = uf.estimate_truncated_power_law_exponent_MLE(interior_areas)
            alpha_estimate_MLE_all = uf.estimate_truncated_power_law_exponent_MLE(all_areas)

        alpha_estimate_LR_all = uf.estimate_power_law_exponent_lr(all_areas)
        alpha_estimate_LR_interior = uf.estimate_power_law_exponent_lr(interior_areas)

        alphahats.append([alpha_estimate_LR_interior[0],alpha_estimate_LR_all[0],alpha_estimate_MLE_interior[0],alpha_estimate_MLE_all[0]])
        alphahaterrors.append([alpha_estimate_LR_interior[1],alpha_estimate_LR_all[1],alpha_estimate_MLE_interior[1],alpha_estimate_MLE_all[1]])

        alpha_estimate_LR_interior = f'${n2str.uncertainty(*alpha_estimate_LR_interior)}$'
        alpha_estimate_MLE_interior = f'${n2str.uncertainty(*alpha_estimate_MLE_interior)}$'
        alpha_estimate_LR_all = f'${n2str.uncertainty(*alpha_estimate_LR_all)}$'
        alpha_estimate_MLE_all = f'${n2str.uncertainty(*alpha_estimate_MLE_all)}$'

        # print(f'Interior estimates:\n   MLE: {} ({two-one:.02f}s)')
        # print(f'    LR: {n2str.uncertainty(*alpha_estimate_LR_interior)} ({three-two:.02f}s)')
        # print(f'All cld estimates:\n   MLE: {n2str.uncertainty(*alpha_estimate_MLE_all)} ({four-three:.02f}s)')
        # print(f'    LR: {n2str.uncertainty(*alpha_estimate_LR_all)} ')
        fit_range = '(${:.0f}\\unit{{km}}$, ${:.0f}\\unit{{km}})$'.format(*range_to_calculate_alpha)

        table_data.append([data.domain_sizes_label[id].replace('domain',''), fit_range,alpha_estimate_LR_interior, alpha_estimate_LR_all, alpha_estimate_MLE_interior, alpha_estimate_MLE_all])

    bias = [f'${2*(small-big)/err:.01f}\\sigma$' for big, small, err in zip(alphahats[0],alphahats[1],alphahaterrors[1])]
    table_data.append([error_label,' ',*bias])

    print(tex_funcs.make_latex_table(table_header, table_data,bold_first=False))

    # Make histogram
    data = DataContainer('GOES', nlattices, False, div_to_save=subarray_id)
    data.interior_areas = np.array(data.interior_areas)
    data.edge_areas = np.array(data.edge_areas)
    bin_edges = np.linspace(*np.log10(range_to_calculate_alpha_subdomain), num=11)
    bin_middles = (bin_edges[1]-bin_edges[0])/2 + bin_edges[:-1]

    interior_counts = np.histogram(np.log10(data.interior_areas), bins=bin_edges)[0]
    edge_counts = np.histogram(np.log10(data.edge_areas), bins=bin_edges)[0]

    fig, ax = plt.subplots(1,1,figsize=single_fig_size)

    ax.scatter(bin_middles, np.log10(interior_counts), s=scatter_size, marker=markers[subarray_id], color=colors[subarray_id])

    ax.scatter(bin_middles, np.log10(interior_counts+edge_counts), s=scatter_size, marker=markers[subarray_id], color=colors[subarray_id], facecolor='none',)

    # Plot regression lines
    (slope, yint), _ = uf.linear_regression(bin_middles, np.log10(interior_counts))
    ax.plot(bin_middles, bin_middles*slope + yint, ls = '--', color=colors[subarray_id])
    (slope, yint), _ = uf.linear_regression(bin_middles, np.log10(interior_counts+edge_counts))
    ax.plot(bin_middles, bin_middles*slope + yint, ls = '--', color=colors[subarray_id])

    ax.text(2.3,3.9,f'$\\hat{{\\alpha}}$ = {alpha_estimate_LR_all}\n(including truncated)', fontdict={'size':font_small})
    ax.text(1.5,2.9,f'$\\hat{{\\alpha}}$ = {alpha_estimate_LR_interior}\n(excluding truncated)', fontdict={'size':font_small})

    ax.set_ylabel('Count')
    ax.set_xlabel('Cloud area, km$^{2}$')

    pf.format_log_log(ax)
    plt.tight_layout()
    pf.savefig('Example bad alpha calc', pdf=True)

def edge_effect_alphas(model='GOES', nlattices=10):

    if model == 'GOES': min_area = 80
    else: min_area = 20
    edge_thresh = 0.5

        
    table_header = ['Domain Size','Fit Range','\\begin{tabular}[c]{@{}l@{}} Excluding \\ Truncated \\\\ LR $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Including \\ Truncated \\\\ LR $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Excluding \\ Truncated \\\\ MLE $\\hat{\\alpha}$\end{tabular}','\\begin{tabular}[c]{@{}l@{}} Including \\ Truncated \\\\ MLE $\\hat{\\alpha}$\end{tabular}']
    table_data = []

    if model == 'Percolation':
        table_data.append(['Exact result 187/91', '-','1.055','1.055','1.055','1.055'])
    
    # First, calculate alpha for the subdomains, then the original domain

    for id in [0, 1,2,3]:

        data = DataContainer(model, n_lattices=nlattices, periodic_boundaries=False, div_to_save=id)
        data.interior_areas = np.array(data.interior_areas)
        data.edge_areas = np.array(data.edge_areas)

        # Remove smallest pixellated clouds
        interior_areas = data.interior_areas[data.interior_areas>min_area] 
        edge_areas = data.edge_areas[data.edge_areas>min_area] 

        # Remove large truncation-affected clouds
        max_area = ef.find_edge_thresh_values(interior_areas, edge_areas, edge_thresh=edge_thresh)
        interior_areas = interior_areas[interior_areas<max_area]
        edge_areas = edge_areas[edge_areas<max_area]

        all_areas = np.append(edge_areas, interior_areas)

        if interior_areas.size==0 or (interior_areas.max()/interior_areas.min())<10:  # then do not estimate alpha
            alpha_estimate_LR_interior = '-'
            alpha_estimate_MLE_interior = '-'
            alpha_estimate_LR_all = '-'
            alpha_estimate_MLE_all = '-'
        else:
            if bootstrap_MLE:
                alpha_estimate_MLE_interior = uf.bootstrap(interior_areas, uf.estimate_truncated_power_law_exponent_MLE, 200)
                alpha_estimate_MLE_all = uf.bootstrap(all_areas, uf.estimate_truncated_power_law_exponent_MLE, 200)
            else:
                print('Incorrect uncertainty for MLE (did not bootstrap)')
                alpha_estimate_MLE_interior = uf.estimate_truncated_power_law_exponent_MLE(interior_areas)
                alpha_estimate_MLE_all = uf.estimate_truncated_power_law_exponent_MLE(all_areas)
            alpha_estimate_LR_interior = uf.estimate_power_law_exponent_lr(interior_areas)
            alpha_estimate_LR_all = uf.estimate_power_law_exponent_lr(all_areas)

            alpha_estimate_LR_interior = f'${n2str.uncertainty(*alpha_estimate_LR_interior)}$'
            alpha_estimate_MLE_interior = f'${n2str.uncertainty(*alpha_estimate_MLE_interior)}$'
            alpha_estimate_LR_all = f'${n2str.uncertainty(*alpha_estimate_LR_all)}$'
            alpha_estimate_MLE_all = f'${n2str.uncertainty(*alpha_estimate_MLE_all)}$'

        # print(f'Interior estimates:\n   MLE: {alpha_estimate_MLE_interior} ({two-one:.02f}s)')
        # print(f'    LR: {alpha_estimate_LR_interior} ({three-two:.02f}s)')
        # print(f'All cld estimates:\n   MLE: {alpha_estimate_MLE_all} ({four-three:.02f}s)')
        # print(f'    LR: {alpha_estimate_LR_all} ')
        fit_range = f'$({min_area}\\mathrm{{km^2}}$, ${max_area:.0f}\\mathrm{{km^2}})$'

        if model == 'Percolation': 
            data.domain_sizes_label[id] = data.domain_sizes_label[id].replace('pixel','site')
            fit_range=fit_range.replace('\mathrm{km^2}', '')

        table_data.append([data.domain_sizes_label[id].replace('domain',''), fit_range,alpha_estimate_LR_interior, alpha_estimate_LR_all, alpha_estimate_MLE_interior, alpha_estimate_MLE_all])

    print('-'*60)
    print(f'--------  {model}  --------')
    print(tex_funcs.make_latex_table(table_header, table_data,bold_first=False))
    print('-'*60)

def edge_effect_plot(model, nlattices=10, wrap=False):

    data = DataContainer(model, nlattices, wrap)


    fig, ax = plt.subplots(1,1,figsize=double_fig_size)

    if model == 'GOES':
        ax.text(1.6,6,'GOES Clouds', size=font_large)
        ax.set_xlim(1.3, 6)
        ax.set_xlabel('Area (km$^2$)')
        ax.set_ylim(0,7)
        ax.text(4.3,4.5,'Including truncated clouds (hollow)', size=font_small)
        ax.text(1.4,1 ,'Excluding truncated\nclouds (filled)', size=font_small)
    elif model == 'Percolation':
        ax.text(1.1,7,'Percolation Clusters', size=font_large)
        ax.set_xlim(0.8, 6)
        ax.set_xlabel('Area (Number of sites)')
        ax.set_ylim(0,8)

        ax.text(4.3,4.5,'Including truncated\nclouds (hollow)', size=font_small)
        ax.text(1.2,1.5 ,'Excluding truncated\nclouds (filled)', size=font_small)

    for i, color, marker in zip(range(len(data.split_divisions)), colors, markers):
        interior_counts = data.interior_counts[i]
        total_counts = 10*( data.interior_counts[i] + data.edge_counts[i]) # offset vertically by factor of 10
        interior_counts[interior_counts<=0] = np.nan
        total_counts[total_counts<=0] = np.nan
        ax.scatter(data.bin_middles, np.log10(interior_counts), color=color, marker=marker, s=scatter_size, label = data.domain_sizes_label[i])
        ax.scatter(data.bin_middles, np.log10(total_counts), color=color, marker=marker, s=scatter_size, facecolor='none', label = '_nolabel')
        # plot edge thresh as vertical line
        edge_thresh = ef.find_edge_thresh_counts(interior_counts, data.edge_counts[i], data.bin_edges)
        if np.isfinite(edge_thresh):
            (slope, yint),_ = uf.linear_regression(data.bin_middles[:10],np.log10(data.interior_counts[2])[:10])
            hist_value_at_edge_thresh = yint+0.6+slope*np.log10(edge_thresh)
            # convert to fig coords
            ymin = (hist_value_at_edge_thresh-1.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ymax = (hist_value_at_edge_thresh+1.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.axvline(np.log10(edge_thresh), ymin=ymin, ymax=ymax, ls='--', color=color)

    ax.legend(ncol=2, loc='upper right', frameon=False, prop={'size': font_small})
    ax.set_ylabel('Count')

    pf.format_log_log(ax)

    plt.tight_layout()
    pf.savefig(f'{model}_Edge_effect_histograms_wrap={wrap}', True, transparent=True)
def edge_v_int_plot(nlattices=10, wrap=False):

    subdomain = 1

    data = DataContainer('GOES', nlattices, wrap)

    color_truncated = '#FA500A'
    color_interior = '#D5E622'
    trunc_mark = 'x'
    int_mark = 'h'

    fig, ax = plt.subplots(1,1,figsize=single_fig_size)

    ax.text(2,4.2,f'GOES {data.domain_sizes_label[subdomain]} subdomains', size=font_small)
    ax.set_xlim(1.5, 6)
    ax.set_xlabel('Area (km$^2$)')
    ax.set_ylim(0,5)
    ax.text(2.5,0.5,'Non-truncated clouds\n(hexagons)', size=font_small)
    ax.text(4.5,2.5 ,'Truncated clouds\n(crosses)', size=font_small)

    # for i, color, marker in zip(range(len(data.split_divisions)), colors, markers):
    interior_counts = data.interior_counts[subdomain]
    edge_counts = data.edge_counts[subdomain]
    interior_counts[interior_counts<=0] = np.nan
    edge_counts[edge_counts<=0] = np.nan
    ax.scatter(data.bin_middles, np.log10(interior_counts), color=color_interior, marker=int_mark, s=scatter_size, label = data.domain_sizes_label[subdomain])
    ax.scatter(data.bin_middles, np.log10(edge_counts), color=color_truncated, marker=trunc_mark, s=scatter_size)
    # plot edge thresh as vertical line
    # edge_thresh = ef.find_edge_thresh_counts(interior_counts, data.edge_counts[subdomain], data.bin_edges)
    # if np.isfinite(edge_thresh):
    #     (slope, yint),_ = uf.linear_regression(data.bin_middles[:10],np.log10(data.interior_counts[2])[:10])
    #     hist_value_at_edge_thresh = yint+0.6+slope*np.log10(edge_thresh)
    #     # convert to fig coords
    #     ymin = (hist_value_at_edge_thresh-1.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
    #     ymax = (hist_value_at_edge_thresh+1.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
    #     ax.axvline(np.log10(edge_thresh), ymin=ymin, ymax=ymax, ls='--', color=colors[subdomain])

    # ax.legend(ncol=2, loc='upper right', frameon=False, prop={'size': font_small})
    ax.set_ylabel('Count')

    pf.format_log_log(ax)

    plt.tight_layout()
    pf.savefig(f'Edge_v_interior_plot_nlattices={nlattices}', True, transparent=True)

def periodic_edge_effect_plot(nlattices=10):

    data = DataContainer('Percolation', nlattices, True)


    fig, ax = plt.subplots(1,1,figsize=double_fig_size)

    ax.text(1.1,6,'Periodic Percolation Clusters', size=font_large)
    ax.set_xlim(0.8, 6)
    ax.set_xlabel('Area (Number of sites)')
    ax.set_ylim(0,7)


    for i, color, marker in zip(range(len(data.split_divisions)), colors, markers):
        interior_counts = data.interior_counts[i]
        total_counts = ( data.interior_counts[i] + data.edge_counts[i])
        # interior_counts[interior_counts<=0] = np.nan
        total_counts[total_counts<=0] = np.nan
        ax.scatter(data.bin_middles, np.log10(total_counts), color=color, marker=marker, s=scatter_size, label = data.domain_sizes_label[i])
        # plot edge thresh as vertical line
        edge_thresh = ef.find_edge_thresh_counts(interior_counts, data.edge_counts[i], data.bin_edges)
        if np.isfinite(edge_thresh):
            (slope, yint),_ = uf.linear_regression(data.bin_middles[:10],np.log10(data.interior_counts[2])[:10])
            hist_value_at_edge_thresh = yint+slope*np.log10(edge_thresh)
            # convert to fig coords
            ymin = (hist_value_at_edge_thresh-1)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ymax = (hist_value_at_edge_thresh+1)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.axvline(np.log10(edge_thresh), ymin=ymin, ymax=ymax, ls='--', color=color)

    ax.set_ylabel('Count')

    pf.format_log_log(ax)
    ax.legend(ncol=2, loc='upper right', frameon=False, prop={'size': font_small})

    plt.tight_layout()
    pf.savefig(f'Periodic_Percolation_Edge_effect_histograms', True, transparent=True)

def exponential_percolation_plot(nlattices=1, wrap=False, log_bin = True):

    
    plot_total = False

    data = ExpDataContainer( nlattices, wrap, log_bin=log_bin)


    fig, ax = plt.subplots(1,1,figsize=double_fig_size)

    if log_bin: ax.text(1.1,7,'Percolation Clusters', size=font_large)
    else: ax.text(75,7,'Percolation Clusters', size=font_large)
    # ax.set_xlim(0.8, 6)
    ax.set_xlabel('Area (Number of sites)')
    ax.set_ylim(0,8)


    for i, color, marker in zip(range(len(data.split_divisions)), colors, markers):
        interior_counts = data.interior_counts[i]
        total_counts = 10*( data.interior_counts[i] + data.edge_counts[i]) # offset vertically by factor of 10
        interior_counts[interior_counts<=0] = np.nan
        total_counts[total_counts<=0] = np.nan
        ax.scatter(data.bin_middles, np.log10(interior_counts), color=color, marker=marker, s=scatter_size, label = data.domain_sizes_label[i])
        if plot_total: 
            ax.scatter(data.bin_middles, np.log10(total_counts), color=color, marker=marker, s=scatter_size, facecolor='none', label = '_nolabel')
        # plot edge thresh as vertical line
        edge_thresh = ef.find_edge_thresh_counts(interior_counts, data.edge_counts[i], data.bin_edges)
        if np.isfinite(edge_thresh):
            (slope, yint),_ = uf.linear_regression(data.bin_middles[:10],np.log10(data.interior_counts[2])[:10])
            hist_value_at_edge_thresh = yint+0.6+slope*np.log10(edge_thresh)
            # convert to fig coords
            ymin = (hist_value_at_edge_thresh-1.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ymax = (hist_value_at_edge_thresh+.5)/(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.axvline(np.log10(edge_thresh), ymin=ymin, ymax=ymax, ls='--', color=color)

    ax.legend(ncol=2, loc='upper right', frameon=False, prop={'size': font_small})
    ax.set_ylabel('Count')

    if log_bin:
        pf.format_log_log(ax)
    else: pf.format_linear_log(ax)

    plt.tight_layout()
    pf.savefig(f'Exponential_histograms_wrap={wrap}_log_bin={log_bin}', True, transparent=True)

main()
