import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotting_functions
from PyThomas import useful_functions as uf
import n2str
from directories import StringDirectories
import powerlaw
import tex_funcs
dirs = StringDirectories()

single_fig_size = (4.5,2.5)     # one rectangular plot
double_fig_size = (6.5,3)
font_small = 8
font_large = 10
capsize = 5
scatter_size = 3
lw = .8
color1 = "#FF4D00"
color2 = "#E8005E"
color3 = "#701EFF"
color4 = "#166AEB"
color5 = "#12F3EA"

def main():

    # estimates_by_sample_size()
    # estimate_accuracy_table()
    estimate_accuracy_plot()    # fig. 2
    histogram_of_counts_in_bin(10,100, 1000, 5000)    # fig. 3
    table_of_counts_in_bin()    # table A1
    # histogram()
    # compare_LR_uncertainties()

class Counts_in_bin(uf.SaveLoad):
    def __init__(self, bin_lower=1, bin_upper=10, n_samples=10000, n_per_sample = 10000, alpha=1):
        min_value = 1

        self.counts_in_bin = []

        self._filename = f'counts_in_bin_{bin_lower}_to{bin_upper}__alpha_{alpha}__{n_samples}_samples_{n_per_sample}_per_sample'
        self._save_to_path = dirs.code_files_directory

        if not self.load_data():
            for i in range(n_samples):
                # Generate power law distributed random variables
                x = powerlaw.Power_Law(xmin=min_value, parameters=[alpha+1]).generate_random(n_per_sample)
                self.counts_in_bin.append(np.count_nonzero((x>bin_lower) & (x<=bin_upper)))

            self.counts_in_bin = np.array(self.counts_in_bin)
            self._is_complete = True
            self.save_data()

class Power_law_LR_estimates(uf.SaveLoad):
    def __init__(self, n_samples=10000, n_per_sample = 10000, alpha=1, min_value = 10, max_value=1000, lin_reg_min_count=30, nbins=30):
        min_value = 10

        self.linreg_estimates = []
        self.linreg_errors = []
        self.smallest_bin_counts = []

        self._filename = f'power_law_LR_estimates__range_{min_value}_to_{max_value}__alpha_{alpha}__{n_samples}_samples_{n_per_sample}_per_sample__lin_reg_min_count_{lin_reg_min_count}_nbins={nbins}'
        self._save_to_path = dirs.code_files_directory

        if not self.load_data():
        # if True:
            for i in range(n_samples):
                # Generate power law distributed random variables
                x = uf.generate_truncated_power_law_randoms(n_per_sample, min_value, max_value, alpha)

                lres, lrer, smallest_bin_count = estimate_power_law_exponent_lr(x,nbins=nbins, min_counts=lin_reg_min_count)

                self.linreg_estimates.append(lres)
                self.linreg_errors.append(lrer)
                self.smallest_bin_counts.append(smallest_bin_count)
                
            self.linreg_estimates = np.array(self.linreg_estimates)
            self.linreg_errors = np.array(self.linreg_errors)
            self.smallest_bin_counts = np.array(self.smallest_bin_counts)
            self._is_complete = True
            self.save_data()
class Power_law_LR_bootstrap(uf.SaveLoad):
    def __init__(self, n_data=[1000, 10000, 100000,1000000], alpha=1, min_value = 10, max_value=1000, lin_reg_min_count=30, nbins=30):
        min_value = 10

        self.linreg_estimates = []
        self.linreg_errors = []

        self._filename = f'power_law_LR_bootstrap__range_{min_value}_to_{max_value}__alpha_{alpha}__{n_data}_data__lin_reg_min_count_{lin_reg_min_count}_nbins={nbins}'
        self._save_to_path = dirs.code_files_directory

        if not self.load_data():
            for n in n_data:
                # Generate power law distributed random variables
                x = uf.generate_truncated_power_law_randoms(n, min_value, max_value, alpha)

                lres, lrer = uf.estimate_power_law_exponent_lr(x,nbins=nbins, min_counts=lin_reg_min_count)


                self.linreg_estimates.append(lres)
                self.linreg_errors.append(lrer)
                    
            self.linreg_estimates = np.array(self.linreg_estimates)
            self.linreg_errors = np.array(self.linreg_errors)
            self._is_complete = True
            self.save_data()

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

        Returns: exponent, uncertainty (95% conf. interval), smallest_bin_count
    """
    data = np.log10(data)
    bin_edges = np.linspace(data.min(), data.max(), nbins+1)
    counts = np.histogram(data, bins=bin_edges)[0].astype(np.float32)
    counts[counts<min_counts] = np.nan

    smallest_bin_count = np.nanmin(counts)

    # Test size of good data
    if np.count_nonzero(np.isfinite(counts)) < 3: 
        return np.nan, np.nan, np.nan
    if max(bin_edges[:-1][np.isfinite(counts)]) - min(bin_edges[:-1][np.isfinite(counts)]) < min_good_data_order: 
        return np.nan, np.nan, np.nan

    (slope, _), (error, _) = uf.linear_regression(bin_edges[:-1], np.log10(counts))

    return -slope, 2*error, smallest_bin_count


def histogram_of_counts_in_bin(bin_lower=1, bin_upper=10, n_samples=1000, n_per_sample = 5000, alpha=1):
    nbins = 30

    filename = f'Histogram of counts between {bin_lower} and {bin_upper}  {n_samples} samples, {n_per_sample} per sample'

    data = Counts_in_bin(bin_lower, bin_upper, n_samples, n_per_sample, alpha).counts_in_bin
    
    log_data = np.log10(data)


    # Bin the counts, calculate gaussian fits
    bin_edges = np.linspace(data.min(), data.max(), nbins+1)
    bin_middles = bin_edges[:-1]+(bin_edges[1]-bin_edges[0])/2
    counts = np.histogram(data, bins=bin_edges, density=False)[0]
    stdev = np.std(data)
    mean = np.mean(data)
    analytical = data.size*(bin_middles[1]-bin_middles[0])/(stdev*np.sqrt(2*np.pi)) * np.exp(-.5*((bin_middles-mean)/stdev)**2) # number of points times deltax times PDF
    print(f'p-value of data following normal distribution: {n2str.scientific_latex(stats.kstest((data-data.mean())/data.std(), stats.norm.cdf).pvalue, 5)}')    # normalize data to a standard normal for comparison with scipy
    # Again for log of data
    log_bin_edges = np.linspace(log_data.min(), log_data.max(), nbins+1)
    log_bin_middles = log_bin_edges[:-1]+(log_bin_edges[1]-log_bin_edges[0])/2
    log_counts = np.histogram(log_data, bins=log_bin_edges, density=False)[0]
    log_stdev = np.std(log_data)
    log_mean = np.mean(log_data)
    log_analytical = data.size*(log_bin_middles[1]-log_bin_middles[0]) /(log_stdev*np.sqrt(2*np.pi)) * np.exp(-.5*((log_bin_middles-log_mean)/log_stdev)**2)
    print(f'p-value of log(data) following normal distribution: {n2str.scientific_latex(stats.kstest((log_data-log_data.mean())/log_data.std(), stats.norm.cdf).pvalue, 5)}')

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=double_fig_size, sharey=True)

    # Set labels for the axes
    ax1.set_xlabel(f'Count $n$ of ${n2str.scientific_latex(bin_lower, 0)} < x_i < {n2str.scientific_latex(bin_upper, 0)}$')
    ax2.set_xlabel(f'$\log_{{10}}(n)$ of ${n2str.scientific_latex(bin_lower, 0)} < x_i < {n2str.scientific_latex(bin_upper, 0)}$')
    # ax1.set_ylabel(f'Frequency of observing a given\ncount of ${n2str.scientific_latex(bin_lower, 0)} < x_i < {n2str.scientific_latex(bin_upper, 0)}$')
    # ax2.set_ylabel(f'Frequency of observing a given\n$\log_{10}$(count) of ${n2str.scientific_latex(bin_lower, 0)} < x_i < {n2str.scientific_latex(bin_upper, 0)}$')
    ax1.set_ylabel(f'Number of observations')
    ax2.set_ylabel('')
    plotting_functions.set_text_size(font_small, font_large)


    ax1.plot(bin_middles, counts, label = "Measured counts", color=color1)
    ax1.plot(bin_middles, analytical, ls='--', label="Gaussian fit", color=color3)
    ax2.plot(log_bin_middles, log_counts, label = "Measured counts", color=color1)
    ax2.plot(log_bin_middles, log_analytical, ls='--', label="Gaussian fit", color=color3)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.tight_layout()
    plotting_functions.savefig(filename, True)

def table_of_counts_in_bin():

    bins = [(9,10),(10,100),(99,100),(100,1000),(1000,10000)]
    n_samples = [1000]
    n_per_samples = [1000,10000,100000,1000000]
    # n_per_samples = [1000,10000,100000]
    alphas = [1,2]

    data_table_header = ['Bin location $i$','Sample size','$\\alpha$','Mean $n_i$','Linear p-value', 'Logarithmic p-value']
    data_table = []

    for bin_lower, bin_upper in bins:
        for n_sample in n_samples:
            for n_per_sample in n_per_samples:
                for alpha in alphas:

                    data = Counts_in_bin(bin_lower, bin_upper, n_sample, n_per_sample, alpha).counts_in_bin
                    
                    log_data = np.log10(data)

                    if bin_lower>=100:
                        bin_tab = f"$\\left({n2str.scientific_latex(bin_lower, 0)}, \ {n2str.scientific_latex(bin_upper, 0)}\\right)$"
                    else:
                        bin_tab = f"$\\left({bin_lower}, \ {bin_upper}\\right)$"
                    mean = f'${n2str.scientific_latex(np.mean(data), 2)}$'
                    n_per_sample_tab = f'${n2str.scientific_latex(n_per_sample, 0)}$'
                    if max(data)<10:
                        # data_table.append([bin_tab,n_per_sample_tab,alpha,'-','-','-'])
                        continue
                    p_lin = stats.kstest((data-data.mean())/data.std(), stats.norm.cdf).pvalue
                    p_log = stats.kstest((log_data-log_data.mean())/log_data.std(), stats.norm.cdf).pvalue
                    if p_lin<0.05:
                        print(f'Lin not normal, mean count = {np.mean(data):.0f}, std count = {np.std(data):.0f}, p={p_lin:.04f}')
                        p_lin = f'$\\mathbf{{{n2str.scientific_latex(p_lin, 2)}}}$'
                    else: p_lin = f'${n2str.scientific_latex(p_lin, 2)}$'
                    if p_log<0.05:
                        print(f'Log not normal, mean count = {np.mean(data):.0f}, std count = {np.std(data):.0f}, p={p_log:.04f}')
                        p_log = f'$\\mathbf{{{n2str.scientific_latex(p_log, 2)}}}$'
                    else: p_log = f'${n2str.scientific_latex(p_log, 2)}$'
                    
                    data_table.append([bin_tab,n_per_sample_tab,alpha,mean,p_lin,p_log])

    print(tex_funcs.make_latex_table(data_table_header, data_table))

def estimates_by_sample_size(max_value=1000, alpha=1, lin_reg_min_count=30): # min value = 10

    lr_sample_sizes = np.geomspace(300,1e4,30)

    lr_est, lr_err = [],[]
    for size in lr_sample_sizes:
        Estimates = Power_law_LR_estimates(1, int(size), max_value=max_value, lin_reg_min_count=lin_reg_min_count, alpha=alpha)
        lr_est.append(Estimates.linreg_estimates[0])
        lr_err.append(Estimates.linreg_errors[0])

    fig, ax = plt.subplots(1,1,figsize=single_fig_size)

    ax.errorbar(lr_sample_sizes, lr_est, yerr=lr_err, color=color4, fmt='o', capsize=capsize,  ms=scatter_size, elinewidth=lw, capthick=lw)

    ax.set_xlabel('Sample size')
    ax.set_ylabel('Estimate of $\\alpha$')

    # True value of alpha
    ax.axhline(alpha, ls='--', color='black', lw=lw)

    ax.set_xscale('log')
    plotting_functions.set_text_size(font_small, font_large)

    plt.tight_layout()
    plotting_functions.savefig(f'estimates_by_sample_size_maxvalue={max_value}_alpha={alpha}_linregmincount={lin_reg_min_count}')

def estimate_accuracy_plot(max_value=1000, alpha=1):

    n_samples = 200
    sample_sizes = [1000,3000,10000]
    # min_counts = [0,10,30,50]
    min_counts = np.arange(start=0,stop=51, step=1)
    nbins = [30,100,300]

    plotted_min_bin_counts = []
    plotted_failure_rates = []

    for min_count in min_counts:
        for size in sample_sizes:
            for nbin in nbins:
                Estimates = Power_law_LR_estimates(n_samples, int(size), max_value=max_value, lin_reg_min_count=min_count, alpha=alpha, nbins=nbin)
                lr_failed = np.count_nonzero((alpha<(Estimates.linreg_estimates-Estimates.linreg_errors)))
                lr_failed += np.count_nonzero((alpha>(Estimates.linreg_estimates+Estimates.linreg_errors)))
                percent_failed = 100*lr_failed/n_samples

                if np.count_nonzero(np.isfinite(Estimates.linreg_estimates)) == 0: continue

                mean_smallest_bin_count = np.mean(Estimates.smallest_bin_counts)
                # if percent_failed<5 and mean_smallest_bin_count<10:
                if nbin == 30 and percent_failed>5:
                    print('-'*50)
                    print(f'Failure rate: {percent_failed:.01f}')
                    print(f'min count: {min_count}, mean smallest bin count: {mean_smallest_bin_count:.01f}')
                    print(f'sample size: {size}, n bins: {nbin}')

                plotted_min_bin_counts.append(min_count)
                # plotted_min_bin_counts.append(mean_smallest_bin_count)
                plotted_failure_rates.append(percent_failed)

    fig, ax = plt.subplots(1,1,figsize=single_fig_size)

    ax.scatter(plotted_min_bin_counts, plotted_failure_rates, s=scatter_size*2, color='black')

    ax.set_ylabel('Failure rate (\%)')
    ax.set_xlabel('Minimum bin count threshold')
    
    ax.set_ylim(-5,105)
    ax.set_xlim(-1,51)

    # Horizontal line displaying accuracy threshold
    ax.axhline(5, ls='--',color='black', lw=lw*0.5, alpha=0.5)

    ax.fill_between([-5,55], 5,120,color='red',alpha=0.5, edgecolor=None)
    ax.text(15, 60, 'Failure rates above 5\% indicate\nunacceptable estimation methods', fontdict={'size':font_small})

    plt.tight_layout()
    plotting_functions.savefig(f'Estimate_accuracy plot, n points = {len(plotted_failure_rates)}', pdf=True)
    
def estimate_accuracy_table(max_value=1000, alpha=1):

    n_samples = 200
    sample_sizes = [1000,3000,10000]
    min_counts = [0,10,30,50]
    nbins = [30,100,300]

    data_table_header = ['Minimum bin count','Sample size','Number bins','Failure rate','Mean $\\hat{\\alpha}$','Mean $\\varepsilon$']
    data_table_1 = []
    data_table_2 = []

    # Lin Reg
    for min_count in min_counts:
        for size in sample_sizes:
            for nbin in nbins:
                Estimates = Power_law_LR_estimates(n_samples, int(size), max_value=max_value, lin_reg_min_count=min_count, alpha=alpha, nbins=nbin)
                lr_failed = np.count_nonzero((alpha<(Estimates.linreg_estimates-Estimates.linreg_errors)))
                lr_failed += np.count_nonzero((alpha>(Estimates.linreg_estimates+Estimates.linreg_errors)))
                percent_failed = 100*lr_failed/n_samples
                if percent_failed>5: percent_failed = f'\\textbf{{{percent_failed:.01f}}}'
                else: percent_failed = f'{percent_failed:.01f}'
                if np.count_nonzero(np.isfinite(Estimates.linreg_estimates)) == 0: percent_failed = 'nan'

                # split into 2 tables
                if min_count in min_counts[:2]:
                    data_table_1.append([min_count, size, nbin,  percent_failed, f'{np.nanmean(Estimates.linreg_estimates):.01f}', f'{np.nanmean(Estimates.linreg_errors):.01f}'])
                else:
                    data_table_2.append([min_count, size, nbin,  percent_failed, f'{np.nanmean(Estimates.linreg_estimates):.01f}', f'{np.nanmean(Estimates.linreg_errors):.01f}'])
                


    print(tex_funcs.make_latex_table(data_table_header, data_table_1, separator="\n", bold_first=False, top_sep='\\tophline',bottom_sep='\\bottomhline', head_sep='\\middlehline').replace('nan','-'))
    print(tex_funcs.make_latex_table(data_table_header, data_table_2, separator="\n", bold_first=False, top_sep='\\tophline',bottom_sep='\\bottomhline', head_sep='\\middlehline').replace('nan','-'))

def compare_LR_uncertainties():
    
    n_data=[1000, 10000, 100000, 1000000]
    alpha=1
    min_value = 10
    max_value=1000
    lin_reg_min_count=30
    nbins=300

    ests = Power_law_LR_bootstrap(n_data=n_data, alpha=alpha, min_value = min_value, max_value=max_value, lin_reg_min_count=lin_reg_min_count, nbins=nbins)

    data_table_header = ['Sample size','$\\hat{{\\alpha}}$','True error','Root mean squared error','Bootstraped uncertainty']
    entries = []
    for i in range(len(n_data)):
        entries.append([n_data[i], f'{ests.linreg_estimates[i]:.03f}', f'{np.abs(alpha-ests.linreg_estimates[i]):.03f}',f'{ests.linreg_errors[i]:.03f}', f'{ests.bootstrap_errors[i]:.03f}'])

    print(f'\\caption{{Comparion of linear regression error estimates. Sample values $x$ drawn from a truncated power law distribution with ${min_value}<x<{max_value}$ with $\\alpha={alpha}$. Bins containing a count of less than {lin_reg_min_count} are omitted from the regression, and {nbins} total bins are used.}}')
    print(tex_funcs.make_latex_table(data_table_header, entries, bold_first=False))

def histogram():
    n = 10000
    min_value = 10
    max_value = 1000
    alpha = 1
    nbins = 300
    min_counts = 100

    x = uf.generate_truncated_power_law_randoms(n, min_value, max_value, alpha)


    data = np.log10(x)
    bin_edges = np.linspace(data.min(), data.max(), nbins+1)
    bin_middles = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])  # shift to center and remove value at end that shifted beyond bins

    counts = np.histogram(data, bins=bin_edges)[0].astype(np.float32)
    counts[counts<min_counts] = np.nan

    fig, ax = plt.subplots(1,1,figsize=single_fig_size)

    ax.scatter(bin_middles, counts)
    plotting_functions.format_log_linear(ax)
    ax.set_yscale('log')
    plotting_functions.set_text_size(font_small, font_large)
    ax.set_xlabel('x')
    ax.set_ylabel('Count')
    ax.set_xlim(np.log10(min_value), np.log10(max_value))
    (slope, _), (unc, _) = uf.linear_regression(bin_middles, np.log10(counts))
    unc = 2*unc
    ax.set_title(f'Estimate ${-slope:.03f}\\pm {unc:.03f}$')
    print(f'Estimate ${-slope:.03f}\\pm {unc:.03f}$')
    accurate = -slope-unc<alpha<-slope+unc
    print(f'Accurate? {accurate}   ({-slope-unc:.03f}, {-slope+unc:.03f})')
    plt.tight_layout()
    plotting_functions.savefig('Histogram of synthetic data', pdf=True)

main()