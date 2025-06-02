# Module with ad-hoc plotting functions to represent results for classification problems.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
# STATUS: completed, functional, and documented.
#
import warnings
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score)

from ..util.validation import (
    checkMultiInputTypes,
)


def confusionMatrix(
        df: pd.DataFrame,
        y_pred: str,
        y_true: str,
        average: str = None,
        y_pred_threshold: float or None = None,
        normalize: bool = True,
        labels: list = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (5, 4),
        dpi: int = 100,
        cmap: str = 'Blues',
        alpha: float = 0.7,
        cm_font_size: int = 14,
        xaxis_label: str = None,
        yaxis_label: str = None,
        axis_label_size: int = 15,
        axis_label_pad: int = 15,
        axis_tick_size: int = 12,
        title: str = '',
        title_size: int = 15,
        title_pad: int = 15,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Function used to represent a confusion matrix from a pandas DataFrame with predictions and true values (e.g.,
    returned by methods :meth:`gojo.core.report.CVReport.getTestPredictions` and
    :meth:`gojo.core.report.CVReport.getTrainPredictions`).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the model predictions.

        Example
        -------
            >>> df
                Out[0]
                                        pred_labels  true_labels
                n_fold indices
                0      2                0.0          0.0
                       6                0.0          0.0
                       11               0.0          0.0
                       12               0.0          0.0
                       13               0.0          0.0
                ...                     ...          ...
                4      987              0.0          0.0
                       992              0.0          0.0
                       1011             0.0          0.0
                       1016             0.0          0.0
                       1018             0.0          0.0

    y_pred : str
        Variable indicating which values are predicted by the model.

    y_true : str
        Variable indicating which values are the ground truth.

    average : str, default=None
        Variable that stratifies the predictions (e.g.n at the folds level) to represent the mean and standard deviation
        values of the confusion matrix.

    y_pred_threshold : float or None, default=None
        Threshold to be used to binarize model predictions.

    normalize : bool, default=True
        Parameter indicating whether to express the normalized confusion matrix (as a percentage).

    labels : list, default=None
        Labels used to identify the classes. By default, they will be C0, C1, ..., CX.

    ax : matplotlib.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(5, 4)
            Figure size.

    dpi : int, default=100
            Figure dpi.

    cmap : str, default='Blues'
        Colormap.

    alpha : float, default=0.7
        Plot opacity.

    cm_font_size : int, default=14
        Confusion matriz font size.

    xaxis_label : str, default=None
        X-axis label.

    yaxis_label : str, default=None
        Y-axis label.

    axis_label_size : int, default=15
        XY-axis label size.

    axis_label_pad : int, default=15
        XY-axis pad.

    axis_tick_size : int, default=12
        XY-ticks size.

    title : str, default=''
        Title.

    title_size : int, default=15
        Title size.

    title_pad : int, default=15
        Title pad.

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
    >>> # ... data loading and model definition
    >>>
    >>> # perform the cross validation
    >>> cv_report = core.evalCrossVal(
    >>>     X=X,
    >>>     y=y,
    >>>     model=model,
    >>>     cv=util.getCrossValObj(cv=5)
    >>> )
    >>>
    >>> # get the model predictions on the test data
    >>> predictions = cv_report.getTestPredictions()
    >>>
    >>> # plot the confusion matrix
    >>> plotting.confusionMatrix(
    >>>     df=predictions,
    >>>     y_pred='pred_labels',
    >>>     y_true='true_labels',
    >>>     average='n_fold',
    >>>     normalize=True,
    >>>     labels=['Class 1', 'Class 2'],
    >>>     title='Confusion matrix',
    >>> )
    >>>
    """
    checkMultiInputTypes(
        ('df', df, [pd.DataFrame]),
        ('y_pred', y_pred, [str]),
        ('y_true', y_true, [str]),
        ('average', average, [str, type(None)]),
        ('y_pred_threshold', y_pred_threshold, [float, type(None)]),
        ('normalize', normalize, [bool]),
        ('labels', labels, [list, type(None)]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('dpi', dpi, [int]),
        ('cmap', cmap, [str]),
        ('alpha', alpha, [float]),
        ('cm_font_size', cm_font_size, [int]),
        ('xaxis_label', xaxis_label, [str, type(None)]),
        ('yaxis_label', yaxis_label, [str, type(None)]),
        ('axis_label_size', axis_label_size, [int]),
        ('axis_label_pad', axis_label_pad, [int]),
        ('axis_tick_size', axis_tick_size, [int]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # avoid inplace modifications when resetting index and applying threshold
    df = df.copy().reset_index()

    if y_pred not in df.columns:
        raise TypeError('Missing "%s" column in dataframe. Available columns are: %r' % (y_pred, list(df.columns)))

    if y_true not in df.columns:
        raise TypeError('Missing "%s" column in dataframe. Available columns are: %r' % (y_true, list(df.columns)))

    # select default parameters
    if xaxis_label is None:
        xaxis_label = 'Predicted label'

    if yaxis_label is None:
        yaxis_label = 'True label'

    font_layout = {
        'family': 'sans-serif',
        'weight': 'normal',
        'size': cm_font_size}

    # binarize input predictions
    if y_pred_threshold is not None:
        df[y_pred] = (df[y_pred] > y_pred_threshold).astype(int)

    # calculate confusion matrices
    cms = []
    if average is not None:
        # calculate a confusion matrix per fold
        if average not in df.columns:
            raise TypeError(
                'Missing "%s" column in dataframe. Available columns are: %r' % (average, list(df.columns)))

        for _, sub_df in df.groupby(average):
            cms.append(confusion_matrix(
                y_true=sub_df[y_true].values,
                y_pred=sub_df[y_pred].values,
                normalize='true' if normalize else None
            ))
    else:
        # calculate a global confusion matrix
        cms.append(confusion_matrix(
            y_true=df[y_true].values,
            y_pred=df[y_pred].values,
            normalize='true' if normalize else None
        ))

    # stack confusion matrices
    cms = np.stack(cms)

    # if the values were normalized represent it as percentages
    if normalize:
        cms = cms * 100

    # format confusion matrix representation
    if average is None:
        assert cms.shape[0] == 1, 'Internal error (0)'
        cms = cms[0]
        avg_cms = cms
        cms_repr = np.empty(shape=cms.shape, dtype=object)
        for i in range(cms.shape[0]):
            for j in range(cms.shape[1]):
                cms_repr[i, j] = ('%.2f' % cms[i, j]) if normalize else ('%d' % cms[i, j])
    else:
        avg_cms = cms.mean(axis=0)
        std_cms = cms.std(axis=0)
        cms_repr = np.empty(shape=cms.shape[1:], dtype=object)
        for i in range(cms_repr.shape[0]):
            for j in range(cms_repr.shape[1]):
                str_val = ('%.2f' % avg_cms[i, j]) + r'$\pm$' + ('%.2f' % std_cms[i, j])
                cms_repr[i, j] = str_val

    # default ticks labels
    if labels is None:
        labels = ['C%d' % (i + 1) for i in range(cms_repr.shape[0])]

    if len(labels) != cms_repr.shape[0]:
        raise TypeError(
            'Number of classes (%d), number of labels provided in the param "labels" (%d)' % (
                cms_repr.shape[0], len(labels)))

    # create figure layout
    with plt.style.context('bmh'):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        # represent the average value
        ax.matshow(avg_cms, cmap=cmap, alpha=alpha)

        # annotate confusion matrix
        for i in range(cms_repr.shape[0]):
            for j in range(cms_repr.shape[1]):
                ax.text(
                    x=j, y=i,
                    s=cms_repr[i, j],
                    va='center', ha='center',
                    fontdict=font_layout)

        # formal layout
        ax.grid(False)
        ax.set_title(title, size=title_size, pad=title_pad)
        ax.set_xlabel(xaxis_label, fontsize=axis_label_size, labelpad=axis_label_pad)
        ax.set_ylabel(yaxis_label, fontsize=axis_label_size, labelpad=axis_label_pad)

        # remove axis-ticks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_yticklabels([''] + labels, fontsize=axis_tick_size)
            ax.set_xticklabels([''] + labels, fontsize=axis_tick_size)

        # change the position of X-axis ticks to the bottom
        ax.xaxis.set_ticks_position('bottom')

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()


def roc(
    df: pd.DataFrame,
    y_pred: str,
    y_true: str,
    average: str = None,
    stratify: str = None,
    n_roc_points: int = 200,
    add_auc_info: bool = True,
    labels: dict = None,
    labels_order: list = None,
    show_random: bool = True,
    random_ls: str = 'dotted',
    random_lw: int or float = 1,
    random_color: str = 'black',
    random_label: str = 'Random',
    ax: mpl.axes.Axes = None,
    figsize: tuple = (5, 4),
    dpi: int = 100,
    style: str = 'ggplot',
    xaxis_label: str = None,
    yaxis_label: str = None,
    lw: float or int or list = None,
    ls: str or list = None,
    colors: list or str = None,
    err_alpha: float = 0.3,
    title: str = '',
    title_size: int or float = 15,
    title_pad: int = 15,
    hide_legend: bool = False,
    legend_pos: str = 'lower right',
    legend_size: int or float = 10,
    xlabel_size: float or int = 13,
    ylabel_size: float or int = 13,
    grid_alpha: float = 0.5,
    save: str = None,
    save_kw: dict = None,
    show: bool = True):
    """ Function used to represent a ROC curve from a pandas DataFrame with predictions and true values (e.g.,
    returned by methods :meth:`gojo.core.report.CVReport.getTestPredictions` and
    :meth:`gojo.core.report.CVReport.getTrainPredictions`).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the model predictions.

        Example
        -------
            >>> df
                Out[0]
                                        pred_labels  true_labels
                n_fold indices
                0      2                0.0          0.0
                       6                0.0          0.0
                       11               0.0          0.0
                       12               0.0          0.0
                       13               0.0          0.0
                ...                     ...          ...
                4      987              0.0          0.0
                       992              0.0          0.0
                       1011             0.0          0.0
                       1016             0.0          0.0
                       1018             0.0          0.0

    y_pred : str
        Variable indicating which values are predicted by the model.

    y_true : str
        Variable indicating which values are the ground truth.

    average : str, default=None
        Variable that stratifies the predictions (e.g.n at the folds level) to represent the mean and standard deviation
        values of the confusion matrix.

    stratify : str, default=None
        Variable used to separate the predictions made by different models.

    n_roc_points : int, default=200
        Number of ROC points to be calculated in order to represent the ROC curve.

    add_auc_info : bool, default=True
        Parameter indicating whether to display the AUC value associated with each model in the legend.

    labels : dict, default=None
        Labels used to identify the models, if not provided the values of the variable specified in `stratify` or a
        default value of "Model" will be used. The labels should be provided as a dictionary where the key will be the 
        value that identifies the model in the input data and the key will be the name given to the model. 

    labels_order : list, default=None
        Order in which the labels will be displayed by default they will be sorted or if parameter `labels` is provided 
        they will appear in the order defined in that input parameter.

    show_random : bool, default=True
        Indicates whether to display the ROC curve associated with a random model.

    random_ls : str, default='dotted'
        Random line style.

    random_lw : int or float, default=1
        Random line width.

    random_color : str, default='black'
        Random line color.

    random_label : str, default='Random'
        Random line label.

    ax : matplotlib.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(5, 4)
            Figure size.

    dpi : int, default=100
            Figure dpi.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    xaxis_label : str, default=None
        X-axis label. Default to "False positive rate"

    yaxis_label : str, default=None
        Y-axis label. Default to "True positive rate"

    lw : float or int or list, default=None
        Line width(s).

    ls : str or list, default=None
        Line style(s).

    colors : list or str, default=None
        Colors used for identifying the dataframe information. A string colormap can be provided.

    err_alpha : float, default=0.3
        Opacity of the error shadow.

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

    legend_size : int, default=12
        Legend size.

    xlabel_size : int, default=13
        Size of the x-label.

    ylabel_size : int, default=13
        Size of the y-label.

    grid_alpha : float, default=0.15
        Gird lines opacity.

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
    >>> # ... model definition and data loading
    >>>
    >>> # train the models
    >>> model1.train(X_train, y_train)
    >>> model2.train(X_train, y_train)
    >>>
    >>> # perform inference on the new data
    >>> y_preds1 = model1.performInference(X_test)
    >>> y_preds2 = model2.performInference(X_test)
    >>>
    >>> # gather the predictions on a single dataframe
    >>> model1_df = pd.DataFrame({
    >>>     'y_pred': y_preds1,
    >>>     'y_true': y_test,
    >>>     'model': ['Model 1'] * y_test.shape[0]
    >>> })
    >>> model2_df = pd.DataFrame({
    >>>     'y_pred': y_preds2,
    >>>     'y_true': y_test,
    >>>     'model': ['Model 2'] * y_test.shape[0]
    >>> })
    >>> model_preds = pd.concat([model1_df, model2_df], axis=0)
    >>>
    >>> # display the ROC curve
    >>> plotting.roc(
    >>>     df=model_preds,
    >>>     y_pred='y_pred',
    >>>     y_true='y_true',
    >>>     stratify='model')
    >>>
    """
    # check input parameters
    checkMultiInputTypes(
        ('df', df, [pd.DataFrame]),
        ('y_pred', y_pred, [str]),
        ('y_true', y_true, [str]),
        ('average', average, [str, type(None)]),
        ('stratify', stratify, [str, type(None)]),
        ('n_roc_points', n_roc_points, [int]),
        ('add_auc_info', add_auc_info, [bool]),
        ('labels', labels, [dict, type(None)]),
        ('labels_order', labels_order, [list, type(None)]),
        ('show_random', show_random, [bool]),
        ('random_ls', random_ls, [str]),
        ('random_lw', random_lw, [int, float]),
        ('random_color', random_color, [str]),
        ('random_label', random_label, [str]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('dpi', dpi, [int]),
        ('style', style, [str]),
        ('xaxis_label', xaxis_label, [str, type(None)]),
        ('yaxis_label', yaxis_label, [str, type(None)]),
        ('lw', lw, [float, int, list, type(None)]),
        ('ls', ls, [str, list, type(None)]),
        ('colors', colors, [list, str, type(None)]),
        ('err_alpha', err_alpha, [float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int]),
        ('hide_legend', hide_legend, [bool]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('xlabel_size', xlabel_size, [float, int]),
        ('ylabel_size', ylabel_size, [float, int]),
        ('grid_alpha', grid_alpha, [float]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]),
    )

    # make a copy of the input dataframe and reset the index
    df = df.copy().reset_index()

    # check variable existence
    if y_pred not in df.columns:
        raise TypeError(
            'Missing "y_pred" variable "%s". Available variables are: %r' % (y_pred, list(df.columns)))

    if y_true not in df.columns:
        raise TypeError(
            'Missing "y_true" variable "%s". Available variables are: %r' % (y_true, list(df.columns)))

    if average is not None:
        if average not in df.columns:
            raise TypeError(
                'Missing "average" variable "%s". Available variables are: %r' % (average, list(df.columns)))

    # select default parameters
    if xaxis_label is None:
        xaxis_label = 'False positive rate'

    if yaxis_label is None:
        yaxis_label = 'True positive rate'

    # extract predictions for individual models
    model_preds = []
    labels_ = []
    if stratify is not None:
        if stratify not in df.columns:
            raise TypeError(
                'Missing "stratify" variable "%s". Available variables are: %r' % (stratify, list(df.columns)))

        for label, preds_df in df.groupby(stratify):
            labels_.append(label)
            model_preds.append(preds_df)

        # sort label order
        if labels is None and not labels_order is None:
            if len(set(labels_order)) != len(labels_order):
                raise ValueError('Duplicated model name in input labels (parameter labels_order) "%r"' % labels_order)
            
            labels = {l: l for l in labels_order}

        # rename and sort the models
        if labels is not None:
            sorted_labels = []
            sorted_preds = []
            for lkey, lval in labels.items():
                for idx, label_ in enumerate(labels_):
                    if label_ == lkey:
                        if lval in sorted_labels:
                            raise ValueError('Duplicated model name "%s"' % lval)
                        sorted_labels.append(lval)
                        sorted_preds.append(model_preds[idx])

            labels_ = sorted_labels
            model_preds = sorted_preds
    else:
        model_preds = [df]
        labels_ = ['Model']

    # select default labels
    labels = labels_

    # check labels shape
    if len(labels) != len(model_preds):
        raise TypeError(
            'Missing labels in "labels". Number of labels provided '\
            '(%d) not match the number of input models (%d)' % (len(labels), len(model_preds)))

    # select line widths
    if lw is None:
        lw = [None] * len(labels)
    elif isinstance(lw, (float, int)):
        lw = [lw] * len(labels)

    if len(lw) != len(labels):
        raise TypeError(
            'Missmatch shape between input models (%d) and "lw" (%d)' % (len(labels), len(lw)))

    # select line styles
    if ls is None:
        ls = ['solid'] * len(labels)
    elif isinstance(ls, str):
        ls = [ls] * len(labels)

    if len(ls) != len(labels):
        raise TypeError(
            'Missmatch shape between input models (%d) and "ls" (%d)' % (len(labels), len(ls)))

    # get colormap colors
    if isinstance(colors, str):
        cmap = plt.get_cmap(colors, len(labels) + 1)
        colors = [mpl.colors.to_hex(cmap(i)) for i in range(len(labels))]

    if not (colors is None or len(labels) == len(colors)):
        raise TypeError(
            'Missmatch shape between input models (%d) and "colors" (%d)' % (len(labels), len(colors)))

    # calculate ROC curves
    roc_data = {}
    xs = np.linspace(0, 1, n_roc_points)
    for model_label, model_df in zip(labels, model_preds):

        # calculate ROC curves averaging the values
        if average is not None:
            all_tpr = []
            aucs = []
            for _, model_df_i in model_df.groupby(average):
                fpr, tpr, _ = roc_curve(model_df_i[y_true].values, model_df_i[y_pred].values)
                mean_tpr = np.interp(xs, fpr, tpr)
                mean_tpr[0] = 0.0
                all_tpr.append(mean_tpr)
                aucs.append(roc_auc_score(
                    y_true=model_df_i[y_true].values, y_score=model_df_i[y_pred].values))

            # aggregated metrics
            mean_tpr = np.mean(all_tpr, axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(all_tpr, axis=0)
            mean_auc = np.mean(aucs, axis=0)

        # calculate a unique ROC curve
        else:
            fpr, tpr, _ = roc_curve(model_df[y_true].values, model_df[y_pred].values)
            mean_tpr = np.interp(xs, fpr, tpr)
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0
            std_tpr = np.zeros_like(mean_tpr)
            mean_auc = roc_auc_score(
                y_true=model_df[y_true].values, y_score=model_df[y_pred].values)

        # save model ROC information
        roc_data[model_label] = {
            'mean': mean_tpr, 'std': std_tpr, 'auc': mean_auc}

    # display the ROC curves
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        for i, (label, roc_model_data) in enumerate(roc_data.items()):
            # select color (if specified)
            color = None if colors is None else colors[i]
            mean_roc = roc_model_data['mean']
            std_roc = roc_model_data['std']

            # add AUC information (if specified)
            if add_auc_info:
                label += ' (AUC=%.2f)' % roc_model_data['auc']

            # plot roc curve
            ax.plot(xs, mean_roc, color=color, lw=lw[i], ls=ls[i], label=label)

            if average is not None:
                ax.fill_between(
                    xs, mean_roc - std_roc, mean_roc + std_roc, color=color, alpha=err_alpha)

        # add legend information
        legend = []
        lines = ax.get_lines()
        for line in lines:
            legend.append(
                mpl.lines.Line2D(
                    [], [], alpha=1.0,
                    color=line.get_color(),
                    lw=line.get_lw(),
                    ls=line.get_ls(),
                    label=line.get_label()))

        # random prediction line
        if show_random:
            ax.plot([0, 1], [0, 1], linestyle=random_ls, lw=random_lw, color=random_color, alpha=1)
            legend.append(
                mpl.lines.Line2D(
                    [], [], color=random_color, lw=random_lw, alpha=1.0, linestyle=random_ls, label=random_label))

        if not hide_legend:
            ax.legend(handles=legend, loc=legend_pos, prop=dict(size=legend_size))

        # figure layout
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.grid(alpha=grid_alpha)
        ax.set_xlabel(xaxis_label, size=xlabel_size)
        ax.set_ylabel(yaxis_label, size=ylabel_size)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()
