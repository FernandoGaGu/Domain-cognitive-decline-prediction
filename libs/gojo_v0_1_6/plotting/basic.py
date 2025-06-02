# Module with basic plotting functions.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)


def linePlot(
        *dfs,
        x: str,
        y: str,
        err: str = None,
        err_alpha: float = 0.3,
        labels: list = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (6, 3.5),
        style: str = 'ggplot',
        dpi: int = 100,
        colors: list or str = None,
        title: str = '',
        title_size: int or float = 15,
        title_pad: int = 15,
        hide_legend: bool = False,
        legend_pos: str = 'upper right',
        legend_size: int or float = 12,
        xlabel_size: float or int = 13,
        ylabel_size: float or int = 13,
        grid_alpha: float = 0.5,
        yvmin: float = None,
        yvmax: float = None,
        xvmin: float = None,
        xvmax: float = None,
        lw: float or int or list = None,
        ls: str or list = None,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Line plot function.

    Parameters
    ----------
    *dfs : pd.DataFrame
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        Y-axis variable. Must be present in the input dataframes.

    err : str
        Variable indicating the errors associated with the lines. Must be present in the input
        dataframes.

    err_alpha : float, default=0.3
        Opacity used to plot the errors.

    labels : list, default=None
        Labels used for identifying the input dataframes.

    ax : matplotlib.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(6, 3.5)
        Figure size.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    dpi : int, default=100
        Figure dpi.

    colors : list or str, default=None
        Colors used for identifying the dataframe information. A string colormap can be provided.

    title : str, default=''
        Plot title.

    title_size : int or float, default=0.5
        Title font size.

    title_pad : int, default=15
        Title pad.

    hide_legend : bool, default=False
        Parameter indicating whether to hide the legend.

    legend_pos : str, default='upper right'
        Legend position.

    legend_size : int, default=12
        Legend size.

    yvmin : float, default=None
        Minimum value in the y-axis.

    yvmax : float, default=None
        Maximum value in the y-axis.

    xvmin : float, default=None
        Minimum value in the x-axis.

    xvmax : float, default=None
        Maximum value in the x-axis.

    xlabel_size : float or int, default=13
        X-axis label size.

    ylabel_size : float ot int, default=13
        Y-axis label size.

    grid_alpha : float, default=0.5
        Grid opacity.

    lw : float or int or list, default=None
        Line(s) width(s).

    ls : str or list, default=None
        Line(s) styles(s).

    save : str, default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict, default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool, default=True
        Parameter indicating whether to save the generated plot.

    Examples
    --------
    >>> from gojo import plotting
    >>>
    >>> # train_info, test_info are pandas dataframes returned by gojo.deepl.fitNeuralNetwork
    >>> plotting.linePlot(
    >>>     train_info, valid_info,
    >>>     x='epoch', y='loss (mean)', err='loss (std)',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model convergence',
    >>>     ls=['solid', 'dashed'],
    >>>     style='default', legend_pos='center right')
    >>>
    """

    checkMultiInputTypes(
        ('x', x, [str]),
        ('y', y, [str]),
        ('err', err, [str, type(None)]),
        ('err_alpha', err_alpha, [float]),
        ('labels', labels, [list, type(None)]),
        ('hide_legend', hide_legend, [bool]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('yvmin', yvmin, [float, type(None)]),
        ('yvmax', yvmax, [float, type(None)]),
        ('xvmin', xvmin, [float, type(None)]),
        ('xvmax', xvmax, [float, type(None)]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('figsize', figsize, [tuple]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('colors', colors, [list, str, type(None)]),
        ('grid_alpha', grid_alpha, [float]),
        ('lw', lw, [list, float, int, type(None)]),
        ('ls', ls, [list, str, type(None)]),
        ('dpi', dpi, [int]),
        ('style', style, [str]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # check input data types
    for i, df in enumerate(dfs):
        checkInputType('df (%d)' % i, df, [pd.DataFrame])
        if x not in df.columns:
            raise TypeError('Missing "x" variable "%s". Available variables are: %r' % (x, list(df.columns)))

        if y not in df.columns:
            raise TypeError('Missing "y" variable "%s". Available variables are: %r' % (y, list(df.columns)))

        if not (err is None or err in df.columns):
            raise TypeError('Missing "err" variable "%s". Available variables are: %r' % (err, list(df.columns)))

    if labels is None:
        labels = ['(%d)' % (i + 1) for i in range(len(dfs))]

    if lw is None:
        lw = [None] * len(dfs)
    elif isinstance(lw, (float, int)):
        lw = [lw] * len(dfs)

    if len(dfs) != len(lw):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "lw" (%d)' % (len(dfs), len(lw)))

    if ls is None:
        ls = ['solid'] * len(dfs)
    elif isinstance(ls, str):
        ls = [ls] * len(dfs)

    if len(dfs) != len(ls):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "ls" (%d)' % (len(dfs), len(ls)))

    if len(dfs) != len(labels):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "legend_labels" (%d)' % (len(dfs), len(labels)))

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(labels) + 1)
        colors = [mpl.colors.to_hex(cmap(i)) for i in range(len(labels))]

    if not (colors is None or len(dfs) == len(colors)):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "colors" (%d)' % (len(dfs), len(colors)))

    # plot information
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        for i, (label, df) in enumerate(zip(labels, dfs)):
            # select color (if specified)
            color = None if colors is None else colors[i]

            # plot line
            ax.plot(
                df[x].values, df[y].values, label=label, lw=lw[i], ls=ls[i], color=color)

            # plot error
            if err is not None:
                ax.fill_between(
                    df[x].values,
                    df[y].values + df[err].values,
                    df[y].values - df[err].values,
                    color=color, alpha=err_alpha)

        # set axis limits
        ax.set_ylim(bottom=yvmin, top=yvmax)
        ax.set_xlim(left=xvmin, right=xvmax)

        # set legend
        if not hide_legend:
            ax.legend(loc=legend_pos, prop=dict(size=legend_size))

        # figure layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=grid_alpha)
        ax.set_xlabel(x, size=xlabel_size)
        ax.set_ylabel(y, size=ylabel_size)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()


def barPlot(
        *dfs,
        x: str,
        y: str,
        labels: list = None,
        colors: list or str = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (6, 3.5),
        style: str = 'ggplot',
        dpi: int = 100,
        err_capsize: float or int = 0.15,
        err_lw: float or int = 1.5,
        grid_alpha: float = 0.15,
        xlabel_size: int or float = 13,
        ylabel_size: int or float = 13,
        title: str = '',
        title_size: int = 15,
        title_pad: int = 15,
        hide_legend: bool = False,
        legend_pos: str = 'upper right',
        legend_bbox_to_anchor: tuple = None,
        legend_size: int = 12,
        yvmin: float = None,
        yvmax: float = None,
        xvmin: float = None,
        xvmax: float = None,
        hide_xlabel: bool = False,
        hide_ylabel: bool = False,
        xaxis_tick_size: int or float = 12,
        yaxis_tick_size: int or float = 12,
        xaxis_rotation: float or int = 0.0,
        yaxis_rotation: float or int = 0.0,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Bar plot function

    Parameters
    ----------
    *dfs
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        Y-axis variable. Must be present in the input dataframes.

    labels : list, default=None
        Labels used for identifying the input dataframes.

    colors : list or str, default=None
        Colors used for identifying the dataframe information. A string colormap can be provided.

    ax : mpl.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(6, 3.5)
        Figure size.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    dpi : int, default=100
        Figure dpi.

    err_capsize : float, default=0.15
        Error capsize.

    err_lw : float, default=1.5
        Error linewidth.

    grid_alpha : float, default=0.15
        Gird lines opacity.

    xlabel_size : int, default=13
        Size of the x-label.

    ylabel_size : int, default=13
        Size of the y-label.

    title : str, default=''
        Plot title.

    title_size : int, default=15
        Title font size.

    title_pad : int, default=15
        Title pad.

    hide_legend : bool, default=False
        Parameter indicating whether to hide the legend.

    legend_pos : str, default='upper right'
        Legend position.

    legend_bbox_to_anchor : tuple, default=None
        Used for modifying the legend position relative to the position defined in `legend_pos`.

    legend_size : int, default=12
        Legend size.

    yvmin : float, default=None
        Minimum value in the y-axis.

    yvmax : float, default=None
        Maximum value in the y-axis.

    xvmin : float, default=None
        Minimum value in the x-axis.

    xvmax : float, default=None
        Maximum value in the x-axis.

    hide_xlabel : bool, default=False
        Parameter indicating whether to hide the x-axis label.

    hide_ylabel : bool, default=False
        Parameter indicating whether to hide the y-axis label.

    xaxis_tick_size : int, default=12
        Controls the x-axis tick size.

    yaxis_tick_size : int, default=12
        Controls the y-axis tick size.

    xaxis_rotation : float or int, default=0.0
        Y-axis tick rotation.

    yaxis_rotation : float or int, default=0.0
        Y-axis tick rotation.

    save : str, default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict, default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool, default=True
        Parameter indicating whether to save the generated plot.


    Examples
    --------
    >>> from gojo import core
    >>> from gojo import plotting
    >>>
    >>> # i.e., compute model performance metrics
    >>> scores_1 = report1.getScores(
    >>>     core.getDefaultMetrics(
    >>>     binary_classification, bin_threshold=0.5))['test']
    >>>
    >>> scores_2 = report1.getScores(
    >>>     core.getDefaultMetrics(
    >>>     binary_classification, bin_threshold=0.5))['test']
    >>>
    >>> # adapt for barplot representation
    >>> scores_1 = scores_1.melt()
    >>> scores_2 = scores_2.melt()
    >>>
    >>>
    >>> plotting.barPlot(
    >>>     scores_1, scores_2,
    >>>     x='variable', y='value',
    >>>     labels=['Model 1', 'Model 2'],
    >>>     title='Cross-validation results'
    >>> )
    """
    checkMultiInputTypes(
        ('x', x, [str]),
        ('y', y, [str]),
        ('labels', labels, [list, type(None)]),
        ('colors', colors, [list, str, type(None)]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('style', style, [str]),
        ('dpi', dpi, [int]),
        ('err_capsize', err_capsize, [float, int]),
        ('err_lw', err_lw, [float, int]),
        ('grid_alpha', grid_alpha, [float]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('hide_legend', hide_legend, [bool]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('legend_bbox_to_anchor', legend_bbox_to_anchor, [tuple, type(None)]),
        ('yvmin', yvmin, [float, type(None)]),
        ('yvmax', yvmax, [float, type(None)]),
        ('xvmin', xvmin, [float, type(None)]),
        ('xvmax', xvmax, [float, type(None)]),
        ('hide_xlabel', hide_xlabel, [bool]),
        ('hide_ylabel', hide_ylabel, [bool]),
        ('xaxis_tick_size', ylabel_size, [int, float]),
        ('yaxis_tick_size', ylabel_size, [int, float]),
        ('xaxis_rotation', xaxis_rotation, [int, float]),
        ('yaxis_rotation', yaxis_rotation, [int, float]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # check input data types
    for i, df in enumerate(dfs):
        checkInputType('df (%d)' % i, df, [pd.DataFrame])
        if x not in df.columns:
            raise TypeError('Missing "x" variable "%s". Available variables are: %r' % (x, list(df.columns)))

        if y not in df.columns:
            raise TypeError('Missing "y" variable "%s". Available variables are: %r' % (y, list(df.columns)))

    if labels is None:
        labels = ['(%d)' % (i + 1) for i in range(len(dfs))]

    # check legend_labels consistency
    if len(labels) != len(dfs):
        raise TypeError(
            'Missing labels in "labels". Number of labels provided ' \
            '(%d) not match the number of input DataFrames (%d)' % (len(labels), len(dfs)))

    if not (colors is None or isinstance(colors, str) or len(dfs) == len(colors)):
        raise TypeError(
            'Mismatch shape between input dataframes (%d) and "colors" (%d)' % (len(dfs), len(colors)))

    # merge dataframes information
    dfs_ = []
    for label, df in zip(labels, dfs):
        df = df.copy()
        df['_label'] = label
        dfs_.append(df)

    merge_df = pd.concat(dfs_, axis=0)

    # plot information
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        sns.barplot(
            data=merge_df, x=x, y=y, hue='_label',
            err_kws={'linewidth': err_lw},
            capsize=err_capsize,
            ax=ax,
            palette=colors)

        # set axis limits
        ax.set_ylim(bottom=yvmin, top=yvmax)
        ax.set_xlim(left=xvmin, right=xvmax)

        # set legend
        if not hide_legend:
            ax.legend(
                loc=legend_pos,
                bbox_to_anchor=legend_bbox_to_anchor,
                prop=dict(size=legend_size))

        # figure layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=grid_alpha)
        ax.set_xlabel('' if hide_xlabel else x, size=xlabel_size)
        ax.set_ylabel('' if hide_ylabel else y, size=ylabel_size)
        ax.set_title(title, size=title_size)
        plt.xticks(fontsize=xaxis_tick_size, rotation=xaxis_rotation)
        plt.yticks(fontsize=yaxis_tick_size, rotation=yaxis_rotation)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()


def scatterPlot(
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str = None,
        hue_mapping: dict = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (6, 4.5),
        style: str = 'ggplot',
        dpi: int = 100,
        maker_size: float or int = None,
        colors: list or str = None,
        title: str = '',
        title_size: int or float = 15,
        title_pad: int = 15,
        hide_legend: bool = False,
        legend_pos: str = None,
        legend_size: int or float = 12,
        xlabel_size: float or int = 13,
        ylabel_size: float or int = 13,
        grid_alpha: float = 0.5,
        yvmin: float = None,
        yvmax: float = None,
        xvmin: float = None,
        xvmax: float = None,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Scatter plot function.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        Y-axis variable. Must be present in the input dataframes.

    hue : str ,default=None
        Hue variable for plotting groups.

    hue_mapping : dict ,default=None
        Hash to map group names from the hue variable in the df to user-defined names.

    ax : mpl.axes.Axes ,default=None
        Axes used to represent the figure.

    figsize : tuple ,default=(6, 4.5)
        Figure size.

    style : str ,default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    dpi : int ,default=100
        Figure dpi.

    maker_size : float or int ,default=None
        Marker size.

    colors : list or str ,default=None
        Colors used for identifying the dataframe information. A string colormap can be provided.

    title : str ,default=''
        Plot title.

    title_size : int or float ,default=15
        Title font size.

    title_pad : int ,default=15
        Title pad.

    hide_legend : bool ,default=False
        Parameter indicating whether to hide the legend.

    legend_pos : str ,default=None
        Legend position.

    legend_size : int or float ,default=12
        Legend size.

    xlabel_size : float or int ,default=13
        X-label size.

    ylabel_size : float or int ,default=13
        Y-label size.

    grid_alpha : float ,default=0.5
        Opcaity of the grid lines.

    yvmin : float ,default=None
        Minimum value in the y-axis.

    yvmax : float ,default=None
        Maximum value in the y-axis.

    xvmin : float ,default=None
        Minimum value in the x-axis.

    xvmax : float ,default=None
        Maximum value in the x-axis.

    save : str ,default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict ,default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool ,default=True
        Parameter indicating whether to save the generated plot.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from gojo import plotting
    >>>
    >>> # load test dataset (Wine)
    >>> wine_dt = datasets.load_wine()
    >>> data = StandardScaler().fit_transform(wine_dt['data'])
    >>> PCs = PCA(n_components=2).fit_transform(data)
    >>> PCs = pd.DataFrame(PCs, columns=['PC1', 'PC2'])
    >>> PCs['target'] = wine_dt['target']
    >>>
    >>> plotting.scatterPlot(
    >>>     df=PCs,
    >>>     x='PC1',
    >>>     y='PC2',
    >>>     hue='target',
    >>>     hue_mapping={0: 'C0', 1: 'C1', 2: 'C2'})
    >>>
    """
    checkMultiInputTypes(
        ('df', df, [pd.DataFrame]),
        ('x', x, [str]),
        ('y', y, [str]),
        ('hue', hue, [str, type(None)]),
        ('hue_mapping', hue_mapping, [dict, type(None)]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('style', style, [str]),
        ('dpi', dpi, [int]),
        ('maker_size', maker_size, [int, float, type(None)]),
        ('colors', colors, [list, str, type(None)]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('hide_legend', hide_legend, [bool]),
        ('legend_pos', legend_pos, [str, type(None)]),
        ('legend_size', legend_size, [int, float]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('grid_alpha', grid_alpha, [float]),
        ('yvmin', yvmin, [float, type(None)]),
        ('yvmax', yvmax, [float, type(None)]),
        ('xvmin', xvmin, [float, type(None)]),
        ('xvmax', xvmax, [float, type(None)]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # check x, y and (optionally) hue variables
    if x not in df.columns:
        raise TypeError('Missing "x" variable "%s". Available variables are: %r' % (x, list(df.columns)))

    if y not in df.columns:
        raise TypeError('Missing "y" variable "%s". Available variables are: %r' % (y, list(df.columns)))

    if hue is not None:
        if hue not in df.columns:
            raise TypeError('Missing "hue" variable "%s". Available variables are: %r' % (hue, list(df.columns)))

    # avoid inplace modifications
    df = df.copy()

    # rename hue if hue_mapping is provided
    if not (hue is None or hue_mapping is None):
        df[hue] = df[hue].apply(lambda v: hue_mapping.get(v, v))

    # get the number of levels
    n_labels = 1
    hue_levels = [None]
    if hue is not None:
        n_labels = len(df[hue].unique())
        hue_levels = df[hue].unique()

    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, n_labels + 1)
        colors = [mpl.colors.to_hex(cmap(i)) for i in range(n_labels)]

    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        for i in range(n_labels):
            # select color (if specified)
            color = None if colors is None else colors[i]

            # separate the data to represent
            if hue is not None:
                df_i = df.loc[df[hue] == hue_levels[i]]
            else:
                df_i = df

            ax.scatter(
                df_i[x].values,
                df_i[y].values,
                label=hue_levels[i],
                s=maker_size,
                color=color)

        # set legend
        if not hide_legend and hue is not None:
            ax.legend(loc=legend_pos, prop=dict(size=legend_size))

        # set axis limits
        ax.set_ylim(bottom=yvmin, top=yvmax)
        ax.set_xlim(left=xvmin, right=xvmax)

        # figure layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=grid_alpha)
        ax.set_xlabel(x, size=xlabel_size)
        ax.set_ylabel(y, size=ylabel_size)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()

