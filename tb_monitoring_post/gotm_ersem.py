"""
# Plotting GOTM-ERSEM output data.
**Author: Jun Sasaki  coded on 2024-09-22  updated on 2024-09-22**<br>
"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def update_datetime_in_file(input_file, start_time, end_time, freq, output_file):
    """
    Replace datetime columns in a new time range in the specified period.
    Purpose: Correct the datetime in input files.

    Parameters
    ----------
    input_file: Original file
    start_time: Start datetime (e.g., '2020-01-01 00:00:00')
    end_time: End datetime (e.g., '2021-01-01 00:00:00')
    freq: Time interval (e.g., '1h' for hourly)
    output_file: New file with updated datetime
    """
    time_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # 元のファイルを読み込む
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 新しいファイルに書き込む
    with open(output_file, 'w') as f:
        for i, line in enumerate(lines):
            # 新しい日時を取得し、既存の行と組み合わせて書き込む
            value = line.split()[2]  # 元のデータの3列目（数値部分）を取得
            new_time = time_range[i]  # 新しい時間
            f.write(f"{new_time} {value}\n")

def get_var(instance, varname):
    """
    Extract varname data in (time, depth) coords.

    Parameters
    ----------
    instance: GotmErsem instance
    varname: Variable name in insatance.ds (xarray.Dataset)

    Returns
    -------
    xarray.DataArray in (z, time)
    """
    return instance.ds[varname].squeeze(dim=['lat', 'lon'], drop=True).T
    
def set_size(w, h, ax=None):
    """ 
    Set figsize from the axes panel size in inches.

    Parameters
    ----------
    w, h: Width and height in inches
    ax: Axes
    """
    if ax is None:
        ax = plt.gca()

    # Get the position of the plot area excluding labels
    pos = ax.get_position()
    plot_width = pos.width
    plot_height = pos.height

    # Subplot parameters related to figure-wide margins (left, right, top, bottom)
    fig_left = ax.figure.subplotpars.left
    fig_right = ax.figure.subplotpars.right
    fig_top = ax.figure.subplotpars.top
    fig_bottom = ax.figure.subplotpars.bottom

    # Calculate the figure width and height based on the plot area
    fig_width = w / plot_width
    fig_height = h / plot_height
    ax.figure.set_size_inches(fig_width, fig_height)

class GotmErsem:
    """
    Read GOTM-ERSEM output netddf into xarray.Dataset.
    """
    def __init__(self, ncfile):
        """
        Parameters
        ----------
        ncfile: GOTM-ERSEM output netcdf file
        """
        self.ncfile = ncfile
        # xarray.Dataset
        self.ds = xr.open_dataset(ncfile)
        # xarray.DataArray
        self.depth = self.ds['z'].squeeze(dim=['lat', 'lon'], drop=True)
        self.time = self.ds['time']
        time = self.time.values
        depth = self.depth.values
        self.time_edges = np.concatenate([time, [time[-1] + (time[-1] - time[-2])]])\
                        - (time[1] - time[0]) / 2
        self.depth_edges = np.concatenate([depth[0, :], [depth[0, -1] + (depth[0, -1]
                         - depth[0, -2])]]) - (depth[0, 1] - depth[0, 0]) / 2
    
class GotmErsemConfig:
    """
    GOTM-ERSEM plot configuration.
    """

    def __init__(self, w, h, dpi=600, bbox_inches='tight', fontsize=12, labelsize=12,
                 title_fontsize=12, xlabel_fontsize=12, ylabel_fontsize=12, linewidth=1,
                 rotation=45):
        '''
        Parameters
        ----------
        w: Width of the axes panel in inches.
        h: Height of the axes panel in inches.
        dpi: Dots per inch.
        bbox_inches: Bbox_inches in savefig.
        fontsize: Font size.
        labelsize: Label size.
        title_fontsize: Title font size.
        xlabel_fontsize: X label font size.
        ylabel_fontsize: Y label font size.
        linewidth: Line width.
        '''
        
        self.w = w
        self.h = h
        self.dpi = dpi
        self.bbox_inches = bbox_inches
        self.fontsize = fontsize
        self.labelsize = labelsize
        self.title_fontsize = title_fontsize
        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel_fontsize = ylabel_fontsize
        self.linewidth = linewidth
        self.rotation = rotation

class GotmErsemData:
    """
    GOTM-ERSEM varname data.
    """
    def __init__(self, instance, varname, xrange=None, yrange=None, vrange=None,
                 xlabel=None, ylabel=None, vlabel=None):
        """
        Parameters
        ----------
        xrange: X range (e.g., ('2020-01-01, '2021-01-01'))
        yrange: Y range (e.g., (-50, 0))
        vrange: Variable range (e.g., (0, 100))
        xlabel: X label
        ylabel: Y label
        vlabel: Variable label
 
        """
        self.instance = instance
        self.varname = varname
        self.xrange = xrange
        self.yrange = yrange
        self.vrange = vrange
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vlabel = vlabel
        setattr(self.instance, varname, get_var(self.instance, varname))

class GotmErsemPlotter:
    """
    GOTM-ERSEM plotter.
    """
    def __init__(self, index, gotm_ersem, plot_cfg, data):
        """
        Parameters
        ----------
        index: Index of the plot
        gotm_ersem: GotmErsem instance
        plot_cfg: GotmErsemConfig instance
        data: GotmErsemData instance
        """
        self.index = index
        self.instance = gotm_ersem
        self.cfg = plot_cfg
        self.data = data
        self.fig, self.ax = plt.subplots()
    
    def make_pcolormesh(self, pcolormesh_kwargs={}, colorbar_kwargs={}):
        """
        Make pcolormesh plot, with defaults overridden by user kwargs.
        """
        # pcolormeshのデフォルト引数
        default_pcolormesh_kwargs = {
            'shading': 'flat',
            'cmap': 'jet',
            'vmin': self.data.vrange[0],
            'vmax': self.data.vrange[1]
        }

        # デフォルトpcolormesh引数をユーザー指定の引数で上書き
        # Overwrite the default pcolormesh argument with a user-specified argument
        default_pcolormesh_kwargs.update(pcolormesh_kwargs)
        mesh = self.ax.pcolormesh(self.instance.time_edges, self.instance.depth_edges, 
                                  getattr(self.instance, self.data.varname), **default_pcolormesh_kwargs)
 
        # yラベル設定
        # y label setting
        self.ax.set_ylabel(self.data.ylabel, fontsize=self.cfg.labelsize)

        # colorbarのデフォルト引数
        # Default arguments for colorbar
        default_colorbar_kwargs = {
            'ax': self.ax,
            'pad': 0.01
        }

        # デフォルトcolorbar引数をユーザー指定の引数で上書き
        # Overwrite the default colorbar argument with a user-specified argument
        default_colorbar_kwargs.update(colorbar_kwargs)
        cbar = self.fig.colorbar(mesh, **default_colorbar_kwargs)
        cbar.set_label(self.data.vlabel, fontsize=self.cfg.labelsize)

        # x軸ラベルの設定
        # x-axis label setting
        self.ax.tick_params(axis='x', rotation=self.cfg.rotation, labelsize=self.cfg.labelsize)
        
        # サイズ設定
        # Size setting
        set_size(self.cfg.w, self.cfg.h, ax=self.ax)

        return self.fig, self.ax
    
    def save2d(self, output_file, **kwargs):
        """
        Save the 2D plot to a file, with kwargs for both pcolormesh and savefig.
        
        Parameters
        ----------
        output_file: str
            Path to save the figure.
        kwargs: dict
            Additional keyword arguments for pcolormesh, colorbar, and savefig.
        """        
        pcolormesh_kwargs = kwargs.get('pcolormesh', {})
        colorbar_kwargs = kwargs.get('colorbar', {})
        
        # pcolormeshとcolorbarに引数を渡す
        self.fig, self.ax = self.make_pcolormesh(pcolormesh_kwargs=pcolormesh_kwargs, 
                                                 colorbar_kwargs=colorbar_kwargs)

        savefig_kwargs = kwargs.get('savefig', {})
        # savefig用のデフォルト引数
        default_savefig_kwargs = {
            'dpi': self.cfg.dpi,
            'bbox_inches': self.cfg.bbox_inches
        }
        # デフォルトsavefig引数をユーザー指定の引数で上書き
        default_savefig_kwargs.update(savefig_kwargs)          
        self.fig.savefig(output_file, **default_savefig_kwargs)
        plt.close(self.fig)
