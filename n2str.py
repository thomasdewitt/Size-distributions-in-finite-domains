"""
    Thomas DeWitt
    Various ways of formatting numbers for displaying on plots etc

"""


import sigfig
from math import log10
import numpy as np

def main():

    # print(integer(9444264953545, scientific=True))
    # print(order_of_magnitude_latex(6*10**8))
    # print(scientific_latex(767894567.783926,20))
    # print(scientific_latex(7678,1))
    # print(scientific_latex(4978,1))
    # print(scientific_latex(2078,1))
    # print(scientific_latex(7078,1))
    # print(scientific_latex(9078,1))
    print(scientific_latex(307894567.783926,0))
    # print(scientific_latex(567894567.783926,0))
    # print(scientific_latex(767894567.783926,3))
    # print(scientific_latex(271358784,1))
    # print(str(scientific_latex(7.654562e12,2)))
    # print('1\\times 10^{{5}}')
    # print(math.log10(5))



def integer(N, scientific=False):
    """
        Takes int and returns (either latex format or with suffix M,B,T for million, billion, trillion):
        9444264953545 -> 9.4T or $9.4 \times 10^{12}$
        3544907332 -> 3.5B or $3.5$\times 10^{9}$
        35449032 -> 35.4M or $3.5$\times 10^{7}$
        3544332 -> 3.5M or $3.5$\times 10^{6}$
        354432 -> 354,432
        3532 -> 3,532
        3544954265454326432645254164326565787976897654876575853642653542541344566764879809878746264032 -> $3.5 \times 10^{93}$
    """
    if int(N) != N:
        raise ValueError('Input must be integer')
    if scientific:
        return '$%s$' % scientific_latex(N, 2)
    else:
        if N > 1000000000000000:
            return '$%s$' % scientific_latex(N, 2)  # do this anyway since it will be impossible to read
        if N > 1000000000000:
            return str(round(N/1000000000000, 1))+'T'
        if N > 1000000000:
            return str(round(N/1000000000, 1))+'B'
        if N > 1000000:
            return str(round(N/1000000, 1))+'M'
    return '{:,}'.format(N)

def scientific_latex(N, sigfigs=3):
    """
        Return N in LaTeX scientific form with specified number of sig figs.

            For sigfigs=0, rounds to the nearest order of magnitude. Note that numbers
            between e.g. 3.16228 and 5 are actually of order 10, since 10^0.5 = 3.16228.
            Similarly, 4000 is of order 10^4.

            print(scientific_latex(767894567.783926,3)) -> 7.68\times 10^{8}
            print(scientific_latex(767894567.783926,1)) -> 8\times 10^{8}
            print(scientific_latex(767894567.783926,0)) -> 10^{9}
            print(scientific_latex(567894567.783926,0)) -> 10^{9}
            print(scientific_latex(367894567.783926,0)) -> 10^{9}
            print(scientific_latex(307894567.783926,0)) -> 10^{8}
    """
    if N is None: return np.nan
    if N == 0: return '0.'+'0'*sigfigs
    if np.any(np.isinf(N)): return '$\\infty$'
    N_log = log10(sigfig.round(N, sigfigs=1))
    O = int(N_log)
    
    num = N/(10**(O))

    if sigfigs == 0: 
        if num>=3.16228:
            return f"10^{{{int(O+1)}}}"
        else:
            return f"10^{{{int(O)}}}"
    elif sigfigs == 1:
        # sigfig.round won't return an int
        string = str(int(sigfig.round(num, decimals=sigfigs-1)))+f'\\times 10^{{{int(O)}}}'
    else:
        string = str(sigfig.round(num, decimals=sigfigs-1))+f'\\times 10^{{{int(O)}}}'

    if O == 0:
        # remove \times 10^{0}
        string = string.split('\\times')[0]

    return string

def uncertainty(number, uncertainty): 
    if number is None or np.isnan(number): return np.nan
    if np.isnan(uncertainty): return number+'\\pm nan'
    if type(uncertainty) == np.ndarray: uncertainty = np.mean(uncertainty)
    if np.isnan(number): return np.nan
    return sigfig.round(number, uncertainty).replace('±','\\pm')

if __name__ == '__main__':
    main()