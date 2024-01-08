"""
    Author: Thomas DeWitt
    Functions for formatting data into LaTeX format.

"""
import n2str



def make_latex_table(header, data, separator = "\n",bold_first = False, top_sep='\n', head_sep='\\hline',bottom_sep='\\hline'):
    """
        Make a formatted latex table out of header and data.

        header: list, Header of table (the header that is visible in the PDF)
        data: list of lists, data that comes after header
        caption: put in latex caption
        reflabel: key for \\label{}
        separator: prints this between each entry in table
        bold_first: bold first entry of each line
        top_sep: Separator between \\begin{tabular}{lll...} and header
        head_sep: Separator between header and data
        bottom_sep: Separator that appears at the bottom of the data

        Returns: LaTeX table, formatted like
        \\begin{tabular}...
            [DATA]
            ...

        \\end{{tabular}}

    """
    

    for i in range(len(data)): 
        if len(header) != len(data[i]): raise ValueError(f'header must be same len as each element of data. data[{i}] was len {len(data[i])} while header is len {len(header)}')
    latex_header = """\n\\begin{tabular}{"""+'l'*len(header)+"""}\n\n""" + top_sep + '\n'
    
    latex_footer = f"""\\end{{tabular}}\n"""
    entries = ''
    # entries += str(header).replace('[','').replace(']','').replace(',',' &')+' \\\\\n'

    for i, entry in enumerate([header,*data]):
        entries += separator
        if bold_first: entries += '\\textbf{'
        entries += str(entry[0])
        if bold_first: entries += '}'
        for thing in entry[1:]:
            entries += ' & '
            entries += str(thing)
        entries += ' \\\\\n'
        if i == 0: entries += '\n'+head_sep+'\n'

    entries += '\n'+bottom_sep+'\n\n'

    return latex_header+entries+latex_footer