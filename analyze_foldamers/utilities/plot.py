import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def plot_distribution(
    types_dict,
    hist_data,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    figure_title=None,
    file_name="angle_distribution.pdf",
    plot_per_page=3,
    marker_string='o-',
    linewidth=1.0,
    markersize=3,
):
    """
    Plot angle or torsion distribution and save to file.
    
    :param types_dict: dictionary mapping angle/torsion numeric type to strings
    :type types_dict: dict{str(int): series_name, ...}
    
    :param hist_data: dictionary containing histogram data
    :type hist_data: dict{series_name_density: 1D numpy array, series_name_bin_centers: 1D numpy array, ...}
    
    :param xlabel: label for x-axis
    :type x_label: str
    
    :param ylabel: label for y-axis
    :type y_label: str
    
    :param xlim: limits for x-axis
    :type xlim: list[xlo, xhi]
    
    :param ylim: limits for y-axis
    :type ylim: list(ylo, yhi)
    
    :param figure_title: title of overall plot
    :type figure_title: str
    
    :param file name: name of file, including pdf extension
    :type file_name: str
    
    :param plot_per_page: number of subplots per pdf page (default=3)
    :type plot_per_page: int
    
    :param marker_string: pyplot format string for line type, color, and symbol type (default = 'o-k')
    :type marker_string: str
    
    :param linewidth: width of plotted line (default=0.5)
    :type linewidth: float
    
    :param markersize: size of plotted markers (default=4 pts)
    :type markersize: float
   
    """
    
    # Determine number of data series:
    # Check if this is a nested dict for multiple files,
    # or a single dict for one file:
    
    outer_key1 = list(hist_data.keys())[0]
    if type(hist_data[outer_key1]) == dict:
        # Plot multiple files on each plot:
        
        nseries = len(types_dict)
        nrow = plot_per_page
        
        # Number of pdf pages
        npage = int(np.ceil(nseries/nrow))
        
        with PdfPages(file_name) as pdf:
            plotted_per_page=0
            page_num=1
            figure = plt.figure(figsize=(8.5,11))
            for key,value in types_dict.items():
                plotted_per_page += 1
                
                plt.subplot(nrow,1,plotted_per_page)
                
                c = 0 # color index
                n_files = len(list(hist_data.keys()))
                
                for file_key, file_value in hist_data.items():
                    # Check if file key is a temperature:
                    if any(c.isalpha() for c in file_key):
                        # use filename as label
                        if len(file_key) > 12:
                            # Truncate long file paths
                            label = "..." + file_key[-12:]
                        else:
                            label = file_key
                    else:
                        label = file_key + " K"
                    color = cm.nipy_spectral(c/n_files)
                    plt.plot(
                        hist_data[file_key][f"{value}_bin_centers"],
                        hist_data[file_key][f"{value}_density"],
                        marker_string,
                        linewidth=linewidth,
                        markersize=markersize,
                        label=label,
                        color=color,
                    )
                    c += 1
                plt.legend(ncol=2,fontsize=6)    
                
                if xlim != None:
                    plt.xlim(xlim[0],xlim[1])
                if ylim != None:
                    plt.ylim(ylim[0],ylim[1])
                
                if xlabel != None:
                    plt.xlabel(xlabel)                
                if ylabel != None:
                    plt.ylabel(ylabel)
                        
                plt.title(f"{types_dict[key]}",fontweight='bold')
                
                if (plotted_per_page >= nrow) or (int(key)==nseries):
                    # Save and close previous page
                    
                    # Use xlabels for bottom row only:
                    # if xlabel != None:
                        # plt.xlabel(xlabel)
                    
                    # Adjust subplot spacing
                    plt.subplots_adjust(hspace=0.3)

                    if figure_title != None:
                        plt.suptitle(f"{figure_title} ({page_num})",fontweight='bold')
                
                    pdf.savefig()
                    plt.close()
                    plotted_per_page = 0
                    page_num += 1
                    if int(key)!= nseries:
                        figure = plt.figure(figsize=(8.5,11))
        
    else:
        # Only one file to plot:
        nseries = len(types_dict)
        nrow = plot_per_page
        
        # Number of pdf pages
        npage = int(np.ceil(nseries/nrow))
        
        with PdfPages(file_name) as pdf:
            plotted_per_page=0
            page_num=1
            figure = plt.figure(figsize=(8.5,11))
            for key,value in types_dict.items():
                plotted_per_page += 1
                
                plt.subplot(nrow,1,plotted_per_page)
                plt.plot(
                    hist_data[f"{value}_bin_centers"],
                    hist_data[f"{value}_density"],
                    marker_string,
                    linewidth=linewidth,
                    markersize=markersize,
                    alpha=0.5,
                )
                
                if xlim != None:
                    plt.xlim(xlim[0],xlim[1])
                if ylim != None:
                    plt.ylim(ylim[0],ylim[1])
                
                if xlabel != None:
                    plt.xlabel(xlabel)
                if ylabel != None:
                    plt.ylabel(ylabel)
                        
                plt.title(f"{types_dict[key]}",fontweight='bold')
                
                if (plotted_per_page >= nrow) or (int(key)==nseries):
                    # Save and close previous page
                    
                    # Use xlabels for bottom row only:
                    # if xlabel != None:
                        # plt.xlabel(xlabel)
                    
                    # Adjust subplot spacing
                    plt.subplots_adjust(hspace=0.3)

                    if figure_title != None:
                        plt.suptitle(f"{figure_title} ({page_num})",fontweight='bold')
                
                    pdf.savefig()
                    plt.close()
                    plotted_per_page = 0
                    page_num += 1
                    if int(key)!= nseries:
                        figure = plt.figure(figsize=(8.5,11))
        
    return
