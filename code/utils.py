
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN



##### GLOBAL DICTIONARY SYMBOLOGIES
# color mappeer for monazite zoning
texture_colors = {'core': '#239BA7', 
                  'overgrowth': '#ECECBB', 
                  'rim': '#E1AA36'}


# color palette for cluster groups (-1 represents outliers with DBSCAN/HDBSCAN)
# age_group_colors = {0:'#f25117', 
#                     1:'#f2a104', 
#                     2:'#00743f', 
#                     3:'#72a2c0', 
#                     4:'#1d65a6', 
#                     5:'#192e5b',
#                     6:'k',
#                     7:'red',
#                     8:'blue',
#                     9:'darkred',
#                     10:'green',
#                     -1:'#b1b8bc'}

age_group_colors = {0:'#ff595e', 
                    1:'#ff924c', 
                    2:'#ffca3a', 
                    3:'#c5ca30', 
                    4:'#8ac926', 
                    5:'#52a675',
                    6:'#1982c4',
                    7:'#4267ac',
                    8:'#6a4c93',
                    9:'#2c4875',
                    10:'#00202e',
                    -1:'#eaeaea'}



def plot_samples_panel(df, ax, color_col, color_map, label_bool=True, timeline=[210, 450], y_label='Age (Ma)'):
    """
    Plot probability density curves for individual monazite samples on a given axis.

    Each row of the DataFrame must contain at least `age` and `error` columns, representing the mean and 1-sigma uncertainty of the sample. The function computes normal distributions around each mean, plots their density curves against sample ages, and styles the axis accordingly.

    Args:
        df (pandas.DataFrame): Input DataFrame containing sample information:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
            - `color_col` (str): Column used to map categories to colors.
        ax (matplotlib.axes.Axes): Matplotlib axis object to draw the panel on.
        color_col (str): Name of the column in `df` used to determine colors.
        color_map (dict): Mapping of values from `color_col` to colors.
        label_bool (bool, optional): Whether to display y-axis labels. Defaults to True.
        timeline (list[int, int], optional): (min_age, max_age) for plotting grid.
            Defaults to (210, 450).
        ylabel (str, optional): Y-axis label text. Defaults to "Age (Ma)".

    Returns:
        float: Maximum density value across all plotted samples, useful for
        scaling ax.
    """
    
    # initialize sample_max to get maximum density
    sample_max = 0.0

    # # copy df & add temporary count column for color_col size (i.e., number of samples in cluster group)
    # df = df.copy()
    # df['_group_n'] = df.groupby(color_col)[color_col].transform('size')
    
    # iterate through samples in df...
    for _, row in df.iterrows():

        # get data...
        mean = row['age']                                           # sample mean
        error = row['error']                                        # sample error (1 sigma)
        symbology = row[color_col]                                  # symbology attribute
        t = np.linspace(mean - 4 * error, mean + 4 * error, 100)    # age range to create PDF (+/- 4 sigma)
        density = norm.pdf(t, mean, error)                          # calculate density curve
        sample_max = max(sample_max, density.max())                 # update sample_max

        # plot curve (y: age vs. x: density)...
        # # if number of samples in group is less than 3 color gray
        # gray_mode = True if row['_group_n'] < 3 else False
        # color = '#eaeaea' if gray_mode else color_map.get(symbology)
        color = color_map.get(symbology)
        ax.fill_between(density, t, color=color, linewidth=0.5, edgecolor='k', alpha=1)
        ax.set_xlim(0, sample_max+0.01)
    
    # adjust ax styling...
    ax.set_xlim(0, sample_max+0.001)                                    # set x limits according to density max
    ax.set_xticks([])                                                   # turn off x ticks
    ax.set_ylim(ax.get_ylim())                                          # explicitly define y ticks (set outside function)
    yticks = np.arange(timeline[0], timeline[1]+1, 10)                  # define y tick marks
    ax.set_yticks(yticks)                                               # set y tick marks
    ax.set_yticklabels([str(i) if i%20 == 0 else '' for i in yticks])   # create y tick labels
    ax.tick_params(axis='y', which='major', labelleft=label_bool)       # turn on/off tick marks
    if label_bool:                                                      # optionally turn on/off y axis label
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(None)
        ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(which='major', axis='y', color='gray', alpha=1, ls="--")    # turn on grid lines

    return sample_max

    


def plot_kde_panel(df, ax, timeline=[210, 450]):
    """
    Plot standard and error-weighted KDEs on a twin-x axis that shares the y-axis (age).

    This function expects `ax` to be a `matplotlib` axis created via `ax_main.twiny()` so it shares the y-limits (age) with your main panel.
    
    It computes:
      1) A standard KDE over sample ages using `scipy.stats.gaussian_kde`.
      2) An error-weighted KDE as a mixture of Gaussians where each sample's
         kernel width equals its 1-sigma error.

    Args:
        df (pandas.DataFrame): Input DataFrame containing sample information:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        ax (matplotlib.axes.Axes): A twin-x axis (e.g., created by `ax_main.twiny()`).
        timeline (list[int, int], optional): (min_age, max_age) for plotting grid.
            Defaults to (210, 450).

    Returns:
        float: The maximum density value across both KDE curves (for scaling/comparison).
    """

    # get info from df...
    ages = df['age'].to_numpy()                        # array of means
    errors = df['error'].to_numpy()                    # array of errors (1 sigma)
    t = np.linspace(timeline[0], timeline[-1], 1000)   # age range to create KDE

    # calculate standard KDE...
    kde = gaussian_kde(ages)
    density_std = kde(t)

    # calculate error-weighted KDE...
    density_err = np.zeros_like(t)
    for age, err in zip(ages, errors):
        density_err += np.exp(-(t-age)**2 / (2*err**2)) / (np.sqrt(2*np.pi)*err)
    density_err /= len(ages)

    # get maximum KDE density value
    kde_max = max(density_std.max(), density_err.max())     

    # plot KDEs on twiny axis...
    ax.plot(density_std, t, color='blue', label='Standard KDE')
    ax.plot(density_err, t, color='red', label='Error-weighted KDE')
    
    # adjust ax styling...
    ax.set_xlim(0, kde_max+0.001)                                             # set x limits of twinx
    ax.set_xticks([])                                                         # turn axes off
    # ax.set_ylim(timeline[0], timeline[1])
    ax.set_ylim(ax.get_ylim())
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)   # turn off y ticks 
    # ax.grid(axis='y', color='gray', alpha=1, ls='--')
    ax.grid(False)                                                            # explicitly turn off grid lines
    ax.set_axis_off()                                                         # turn off axes

    return kde_max



def plot_yttrium_panel(df, ax, y_col='Y2O3', y_color_col='garnet'):
    """
    Plot continuous geochemical values against age, symbolized by a binary column.

    The function separates the DataFrame into two groups based on `y_color_col` (e.g., garnet-present vs. garnet-absent) and plots each with distinct colors/symbols. X-axis is the continuous variable (`y_col`), and y-axis is always sample `age`.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - `y_col` (float): Continuous values to plot on the x-axis.
            - `y_color_col` (int or bool): Binary indicator column
              (1 = red symbol, 0 = blue symbol).
        ax (matplotlib.axes.Axes): Matplotlib axis to draw the panel on.
        y_col (str, optional): Column name of the continuous variable to plot on
            the x-axis. Defaults to "Y2O3".
        y_color_col (str, optional): Column name of the binary indicator used
            for symbolizing points. Defaults to "garnet".

    Returns:
        None: The function modifies the given `ax` in place.
    """

    # separate y_col values (Y2O3) by y_color_col (garnet indicator)
    grt = df[df[y_color_col] == 1].copy()
    no_grt = df[df[y_color_col] == 0].copy()

    # plot points
    ax.plot(grt[y_col], grt['age'], marker='o', markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.3, linewidth=0, alpha=0.5)
    ax.plot(no_grt[y_col], no_grt['age'], marker='o', markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.3, linewidth=0, alpha=0.5)

    # adjust ax styling...
    xlims = (df[y_col].min(), df[y_col].max())                                 # get min and max limits
    ax.set_xlim(xlims[0] - 0.3, xlims[1] + 0.3)                                # set x limits to min and max +/- buffer
    ax.set_xticks([df[y_col].min(), df[y_col].max()])                          # set ticks to min and max

    ax.set_xticklabels([round(xlims[0], 2), round(xlims[1], 2)], fontsize=8)   # show min and max ticks
    ax.set_xlabel(f"{y_col} (Wt. %)", fontsize=8)                              # show x axis label

    ax.set_ylim(ax.get_ylim())                                                 # set y limit to parent
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)    # minimize tick marks
    ax.grid(axis='y', color='gray', alpha=1, ls='--')
    # ax.grid(False)                                                             # turn off grid lines




def plot_ageinterp_panel(df, ax, color_col, color_map, bar_width=0.4, gap=0.3, ci='2sigma'):
    """
    Plot weighted mean age metrics (individual ages, errors, and MSWD).

    If at least three ages are present, this function:
        1. Computes the weighted mean age and confidence interval for the group.
        2. Plots a semi-transparent 95% CI ribbon around the mean.
        3. Plots vertical bars for each individual sample age with +/-1 sigma error, colored by group layer (color_col).
        3. Overlays a horizontal line at the weighted mean age.

    If less than three ages are present, this function:
        1. Plots gray vertical bars for each individual sample age with +/-1 sigma error. 

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        ax (matplotlib.axes.Axes): Axis object to plot on.
        color_col (str): Name of the category (key in `color_map`) used to
            determine plotting color.
        color_map (dict): Mapping of categories to matplotlib colors.
        bar_width (float, optional): Width of the individual age bars.
            Defaults to global 0.4.
        gap (float, optional): Horizontal gap between bars. Defaults to 0.3.
        ci (str, optional): Confidence interval to use for weighted mean ('2sigma' or '95%'). Defaults to '2sigma'.

    Returns:
        dict or None: Weighted mean statistics dictionary (from `cluster_stats`) if
        the group has 3 or more ages; otherwise, returns None.
    """

    # pre-proceessing...
    df = df.sort_values('age').copy()                 # sort df by age (ascending)
    color_col = color_col if len(df) >= 3 else -1     # enforce gray for undersized groups
    ages = df['age'].to_numpy()                       # extract age array
    errors = df['error'].to_numpy()                   # extract error array (1 sigma SE)

    # calculate weighted mean age and stats...   
    stats = cluster_stats(ages, errors, ci=ci)

    # plot weighted mean CI ribbon for groups with at least 3 ages (on bottom)...
    if stats['n'] >= 3:
        y0 = stats['mean'] - stats['halfwidth']
        y1 = stats['mean'] + stats['halfwidth']
        ax.axhspan(y0, y1, xmin=0, xmax=1, facecolor=color_map[color_col], alpha=0.35, edgecolor='none')

    # plot age/SE bars for each group...
    xstart = gap * 2
    for age, err in zip(ages, errors):
        bar = Rectangle(xy=(xstart, age-err), width=bar_width, height=err*2, facecolor=color_map[color_col], edgecolor='k', linewidth=0.4)
        ax.add_patch(bar)
        xstart += bar_width + gap
    
    # plot weighted mean line...
    if stats['n'] >= 3:
        ax.axhline(y=stats['mean'], xmin=0, xmax=1, color='k', linewidth=1)

    # stylize axis...
    ax.set_xticks([])                                                         # turn off x ticks
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)   # minimize y axis ticks
    ax.grid(axis='y', color='gray', alpha=1, ls='--')                         # show grid lines
    ax.set_ylim(ax.get_ylim())                                                # explicitly set y axis limits from parent

    if stats['n'] >= 3:
        return stats




def cluster_stats(ages, se, ci='2sigma'):
    """
    Compute Isoplot-style weighted mean statistics for a set of ages.

    The weighted mean uses inverse-variance weights (w_i = 1 / sigma_i^2). The mean
    uncertainty is inflated by sqrt(MSWD) when MSWD > 1, matching common
    geochronology practice. You can report either a 2 sigma half-width or a 95% CI
    half-width (Student's t, df = n-1).

    Args:
        ages (array-like of float): Measured ages (units arbitrary but consistent).
        se (array-like of float): 1 sigma uncertainties (standard errors) for each age.
        ci (str, optional): If '2sigma', return half-width = 2 sigma_mean.
            If '95%', return half-width for a two-sided 95% confidence interval
            using t_{0.975, n-1}. Defaults to '2sigma'.

    Returns:
        dict: {
            'mean' (float): Weighted mean age.
            'mswd' (float): Mean square of weighted deviates, chi squared/(n-1).
            'n' (int): Number of observations.
            'sigma_mean' (float): Standard error of the weighted mean, inflated
                                 by sqrt(MSWD) if MSWD > 1.
            'halfwidth' (float): Requested half-width (2 sigma or 95% CI).
        }
    """

    # from data...
    n = len(ages)           # number of observations
    w = 1.0 / (se ** 2)     # weights (inverse variance)
    

    # weighted mean
    W = np.sum(w)                       # sum of inverse variances
    mu = np.average(ages, weights=w)    # weighted mean

    # MSWD (chi squared/(n-1))...
    chi2 = np.dot(w, (ages - mu) ** 2)    # chi squared statistic
    df_ = max(n-1, 1)                     # degrees of freedom
    mswd = chi2 / df_                     # mswd

    # compute uncertainty on the weighted mean (inflate if MSWD>1)...
    sigma_mu = np.sqrt(1.0 / W)                                              # uninflated uncertainty
    sigma_mu_adj = sigma_mu * np.sqrt(mswd) if mswd > 1 else sigma_mu        # inflate if mswd > 1

    # calculate 2 sigma or 95% CI of mean...
    if ci == '2sigma':
        half_width = 2.0 * sigma_mu_adj

    elif ci == '95%':
        tcrit = t.ppf(0.975, df=df_)          # calculate student's t critical value for 95% CI
        half_width = tcrit * sigma_mu_adj     # get halfwidth

    return {
        "mean": mu,                    # weighted mean age
        "mswd": mswd,                  # mswd
        "n": n,                        # number of observations
        "sigma_mean": sigma_mu_adj,    # standard error of the weighted mean (1 sigma)
        "halfwidth": half_width        # confidence interval halfwidth (2 sigma or 95% CI)
    }




def horizontal_stack(ax, k_label, y_pos, counts_dict, color_map, second_arr):
    """
    Render a label, a 0-1 stacked percentage bar, and an overlaid horizontal boxplot. This overlay approach avoids the small pixel misalignments seen with `twiny()`.

    This draws two visual layers that share the same y-space:
    1. On the base axis `ax` (x-scale = 0-1), it places a label at `y_pos`
        and a stacked bar just below it whose segment widths are the category
        percentages from `counts_dict` and colors from `color_map`.
    2. On a transparent overlay axis (created once as `ax._box_overlay` and
        reused), it draws a horizontal boxplot of `second_arr` using an independent
        x-scale (e.g., Y2O3 values) while keeping y aligned with the base axis.
    
    Args:
        ax (matplotlib.axes.Axes): Parent axis on which the stacked bar and label
            are drawn. An overlay axis is created/reused and stored as `ax._box_overlay`.
            The base axis x-limits are set to [-0.1, 1.1] and x-ticks are hidden.
        k_label (int): Zero-based cluster index; displayed as "Cluster {k_label+1}".
        y_pos (float): Baseline y position (in data units) for the label. The stacked
            bar is drawn from y ≈ `y_pos+12` down to y ≈ `y_pos+2`. The boxplot is
            centered above them at y ≈ `y_pos+19` (clamped to the panel's y-limits).
        counts_dict (Mapping[str, int]): Mapping of category:count used to compute
            segment percentages for the stacked bar (width = count/total).
        color_map (Mapping[str, color]): Mapping of category:matplotlib color used
            for the stacked bar segments.
        second_arr (array-like of float): Continuous values to summarize with a
            horizontal boxplot on the overlay axis (e.g., Y2O3).

    Returns:
        None: The function modifies `ax` (and its cached overlay axis) in place.
    """

    ##### stacked bar chart
    # format axis 1 for stacked bar chart...
    ax.set_xlim(-0.1, 1.1)                                                    # set limits
    ax.set_xticks([])                                                         # remove x ticks
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)   # minimize tick marks
    ax.grid(False)                                                            # turn off grid

    # plot cluster name at y position...
    cluster_label = f"Cluster {k_label+1}"                                # create string label (group+1 not start at 0)
    ax.text(x=0, y=y_pos, s=cluster_label, fontsize=7, weight='bold')     # plot text

    # plot stacked bar chart...
    total = sum(counts_dict.values())                                      # sum total samples in dictionary
    x0 = 0                                                                 # start position of stacked bar
    
    # iterate through dictionary...
    for tex, count in counts_dict.items():
        x1 = count / total                                                 # end position of bar part (percent of total)
        count_bar = Rectangle(xy=(x0, y_pos+12), width=x1, height=-10,     # lower left start position, horizontal width, vertical height
                              facecolor=color_map[tex],                    # map to dictionary for clusters/components colors
                              linewidth=0.5, edgecolor='k')                # outline
        ax.add_patch(count_bar)                                            # add bar part below text
        x0 += x1                                                           # update start position for next bar part


    ##### setup axis for boxplot
    # create transparent overlay on top of ax...
    # NOTE: for some reason using ax.twiny was creating slight mismatch on y axis (annoying but not signficant) - this fixes it
    # setup custom attribute to ax first time...
    if not hasattr(ax, "_box_overlay"):
        box_ax = ax.figure.add_axes(ax.get_position(), frameon=False)   # same rectangle as ax (no sharing)
        box_ax.set_yticks([])                                           # turn off y ticks
        box_ax.set_xticks([])                                           # turn off x ticks
        box_ax.patch.set_alpha(0.0)                                     # make overlay transparent (important)
        box_ax.set_zorder(ax.get_zorder() + 1)                          # draw on top of ax
        ax._box_overlay = box_ax                                        # stash for re-use
    
    # update custom attribute with new positioning on the new parent ax...
    else:
        box_ax = ax._box_overlay
        box_ax.set_position(ax.get_position())                          # keep aligned if layout moved ax

    # hides spines/ticks/labels; the patch stays, but we set it transparent above
    box_ax.set_axis_off()

    # keep overlay in sync with parent limits but NOT shared
    box_ax.set_ylim(ax.get_ylim())                     # same y-space as ax parent
    box_ax.set_xlim(-0.1, np.max(second_arr) + 0.2)    # different x-space

    # clamp to stay within y-lims...
    ylo, yhi = box_ax.get_ylim()
    pos = max(min(y_pos + 19, max(ylo, yhi) - 1), min(ylo, yhi) + 1)


    ##### boxplot
    # normalize second_arr to [0, 1]
    # scaled_arr = (second_arr - np.min(second_arr)) / (np.max(second_arr) - np.min(second_arr))

    if len(second_arr) == 1:
        scaled_arr = second_arr.astype(float)   # keep actual value
        # box_ax.set_xlim(second_arr[0] - 0.5, second_arr[0] + 0.5)  # give it a little space
    else:
        rng = np.ptp(second_arr)
        scaled_arr = (second_arr - np.min(second_arr)) / rng
        # box_ax.set_xlim(0, 1)

    # draw the boxplot on the overlay axis
    # box_ax.boxplot(second_arr, positions=[pos], vert=False, widths=10, patch_artist=True,
    #     boxprops=dict(facecolor='tab:blue', color='k', linewidth=0.5),
    #     medianprops=dict(color='k', linewidth=0.5),
    #     whiskerprops=dict(color='k', linewidth=0.5),
    #     capprops=dict(color='k', linewidth=0.5),
    #     flierprops=dict(color='k', markersize=2, linewidth=0.25, markerfacecolor='k')
    # )
    box_ax.boxplot(scaled_arr, positions=[pos], vert=False, widths=10, patch_artist=True,
        boxprops=dict(facecolor='tab:blue', color='k', linewidth=0.5),
        medianprops=dict(color='k', linewidth=0.5),
        whiskerprops=dict(color='k', linewidth=0.5),
        capprops=dict(color='k', linewidth=0.5),
        flierprops=dict(color='k', markersize=2, linewidth=0.25, markerfacecolor='k')
    )

    # set overlay axis to 0–1 for x, but keep same y-lims as parent
    box_ax.set_xlim(-0.1, 1.1)
    box_ax.set_ylim(ax.get_ylim())
    




def kmeans_processing(df, k, random_state=111):
    """
    Cluster ages using KMeans and remap labels in chronological order.

    This function applies KMeans clustering to the 'age' column of the DataFrame, remaps cluster labels so that younger mean ages have lower
    labels, and sorts the DataFrame by cluster label and error. The index is reset before returning.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        k (int): Number of clusters to compute.
        random_state (int, optional): Random seed for reproducibility. 
            Defaults to 111.

    Returns:
        pandas.DataFrame: DataFrame with an added 'labels' column (int)
        for chronological cluster assignment, sorted by cluster label
        and error, with index reset.
    """

    # extract age array
    ages = df['age'].to_numpy().reshape(-1,1)

    # fit K-means model
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto').fit(ages) 

    # update df...
    df['labels'] = kmeans.labels_                                 # attach cluster labels
    means = df.groupby('labels')['age'].mean().sort_values()      # calculate cluster means and sort (ascending)
    remap = {old: new for new, old in enumerate(means.index)}     # dict to re-map labels in chronological order
    df['labels'] = df['labels'].map(remap).astype(int)            # update new cluster labels
    df.sort_values(['labels', 'error'], inplace=True)             # sort by cluster labels then errors
    df.reset_index(drop=True, inplace=True)                       # reset df index

    return df





def gmm_processing(df, k, random_state=111, timeline=[210, 450]):
    """Cluster ages with a Gaussian Mixture, remap labels by mean age, and sort the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        k (int): Number of mixture components.
        random_state (int, optional): Seed for reproducibility. Defaults to 111.
        timeline (list[int, int], optional): (min_age, max_age) for plotting grid.
            Defaults to (210, 450).

    Returns:
        tuple:
            pandas.DataFrame: With chronological 'labels' (0 = youngest), sorted by
                ['labels', 'error'] and index reset.
            list[np.ndarray]: Weighted component PDFs evaluated on the plotting grid.
            np.ndarray: Plotting grid (ages) used to evaluate PDFs.
            np.ndarray: Mixture PDF evaluated on the plotting grid.
    """

    ##### GMM MODEL
    # extract age array
    ages = df['age'].to_numpy().reshape(-1, 1)

    # fit GMM model
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=10, random_state=random_state).fit(ages)

    # extract fitted component parameters...
    means = gmm.means_.ravel()                                               # means of each component
    stds  = np.sqrt(gmm.covariances_.reshape(-1)).ravel()                    # standard deviations of each component
    weights = gmm.weights_.ravel()                                           # weights of each component
    order = np.argsort(means)                                                # get order of means
    means, stds, weights = means[order], stds[order], weights[order]         # re-order information

    # responsibilities & labels mapped to re-ordered components...
    resp = gmm.predict_proba(ages)[:, order]                                 # sample responsibilities mapped to new labels; [samp., comp.]
    labels = resp.argmax(axis=1)                                             # final labels assigned to maximum component responsibility

    # update df...
    df['labels'] = labels
    df.sort_values(['labels', 'error'], inplace=True)
    df.reset_index(drop=True, inplace=True)


    ##### PLOT PRODUCTS
    # build timeline for plotting
    t = np.linspace(timeline[0], timeline[1], 1000)

    # get weighted component PDFs...
    stds_plot = np.clip(stds, 0.5, None)                               # plotting-only guard against extremely thin curve
    components = [w * norm.pdf(t, loc=mu, scale=sd)             # calculate PDFs and keep in list
                  for mu, sd, w in zip(means, stds_plot, weights)]    
    mixture = np.sum(components, axis=0)                               # total mixture curve

    return df, components, t, mixture




def plot_components_panel(ax, x_components, y_timeline):
    """Plot GMM component density curves against age on a twin x-axis.

    Plots each component's density (x) versus age (y) on the provided Matplotlib Axes (typically created via ``ax.twiny()``). Styles the
    panel to be annotation-like (no ticks/axis frame).

    Args:
        ax (matplotlib.axes.Axes): Target axes (usually ``twiny()`` of a main axis).
        x_components (Sequence[np.ndarray]): Iterable of 1D arrays, each the
            density values for a single GMM component, sampled along ``y_timeline``.
            All arrays must be the same length as ``y_timeline``.
        y_timeline (np.ndarray): 1D array of ages (e.g., Ma) corresponding to
            each column of ``x_components``.

    Returns:
        matplotlib.axes.Axes: The input axes after plotting and styling.
    """

    # plot age vs. component density curves...
    for _, c in enumerate(x_components):
        ax.plot(c, y_timeline, color='k', linewidth=1.5, linestyle='--')

    # stylize axes...
    ax.set_xlim(0, np.max(x_components))                                       # set x limits
    ax.set_xticks([])                                                          # turn off x tick marks
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)    # minimize y tick marks
    ax.grid(False)                                                             # explicitly turn off grid lines
    ax.set_axis_off()                                                          # turn off axis lines




def dbscan_processing(df, eps, min_samples):
    """
    Cluster ages using DBSCAN and remap labels chronologically.

    Applies DBSCAN to the 'age' column of the input DataFrame, assigns cluster labels, and remaps them in chronological order based on
    mean age (youngest cluster = 0, oldest cluster = K-1). Noise points (-1 from DBSCAN) are temporarily mapped to the end for sorting, then
    restored to -1 so they appear last in the sorted DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        eps (float): The DBSCAN neighborhood radius.
        min_samples (int): Minimum number of samples required in a 
            neighborhood to form a core point.

    Returns:
        pandas.DataFrame: DataFrame with a new column:
            - 'labels' (int): chronological cluster assignments 
              (0 = youngest, K-1 = oldest, -1 = noise).
        The DataFrame is sorted by label (with noise last) and then by
        'error', with the index reset.
    """

    ##### DBSCAN MODEL
    # extract age array
    ages = df['age'].to_numpy().reshape(-1, 1)

    # fit DBSCAN model
    dbs = DBSCAN(eps=eps, min_samples=min_samples).fit(ages)


    ##### UPDATE DF...
    # attach cluster labels
    df['labels'] = dbs.labels_

    # calculate mean age per cluster (exclude noise = -1)
    means = (df.loc[df['labels'] != -1].groupby('labels')['age'].mean().sort_values())

    # calculate total number of real clusters (not noise)
    K = len(means)

    # build mapper between original and new labels...
    remap = {old: new for new, old in enumerate(means.index)}    # map to real clusters
    remap[-1] = K                                                # map to noise (highest value)
    df['labels'] = df['labels'].map(remap).astype(int)           # re-map labels & keep noise as -1

    df.sort_values(['labels', 'error'], inplace=True)            # sort by label order (ascending) and then error
    df.reset_index(drop=True, inplace=True)                      # reset index
    df.loc[df['labels']==K, 'labels'] = -1                       # replace noise back with -1

    return df




def relative_drop_kmeans_k(wcss, start_k=2, min_gain=0.05, alpha=0.9, lookahead=2):
    """
    Choose the optimal number of clusters (k) from a sequence of WCSS values using a relative-drop + slowdown heuristic.

    Parameters
    ----------
    wcss (array-like, float): 
        Within-cluster sum of squares for cluster counts k = 1..N.
    min_gain (float, default=0.05):
        Minimum required relative improvement (fractional drop) for a step to be considered as the elbow.
    alpha (float, default=0.6):
        Slowdown factor. After the candidate step, the next 1..`lookahead` drops must each be <= alpha * current drop.
    lookahead (int, default=2):
        Number of subsequent steps to check for slowdown. If 1, only check the immediate next drop; if 2, check the next two.

    Returns
    -------
    k_star : int
        The optimal cluster number (arrival k). If no step passes the heuristic, the function falls back to the arrival k corresponding to the maximum relative drop.

    Notes
    -----
    - Relative drop at step k->k+1 is defined as:
        (WCSS[k-1] - WCSS[k]) / WCSS[k-1]
    - The "arrival k" is the cluster number after taking the drop
      (i.e., k+1 in the notation above).
    - Example: if the biggest qualifying drop is from 4->5 clusters,
      the function returns k_star = 5.
    """

    ##### pre-processing...
    wcss = np.asarray(wcss, dtype=float)             # convert to array of floats
    rel_drop = (wcss[:-1] - wcss[1:]) / wcss[:-1]    # calculate relative drops of each step


    ##### find optimal k (k_star)...
     # initialize k_star
    k_star = None 

    # iterate through each step; i indexes step (k=i -> i+1)...
    for i in range(1, len(rel_drop)):

        # get current relative drop
        curr = rel_drop[i]

        # check if current relative drop is NAN or less than the minimum gain threshold; return None if true
        if not np.isfinite(curr) or curr < min_gain:
            continue

        # determine how many future steps are available to actually check (near end of array there won't be enough steps left)
        remaining_steps = len(rel_drop) - 1 - i
        steps_to_check = min(lookahead, remaining_steps)

        # check if enough steps remain for slowdown calculation; return None if not
        if steps_to_check <= 0:
            continue
        
        # get next drops to compare slowdown to current drop
        next_drops = rel_drop[i+1 : i+1+steps_to_check]

        # check if slowdown calculation is validarrival k (since rel_drop[i] is step i+1 -> i+2)
        ok = all((not np.isfinite(nd)) or (nd <= alpha * curr) for nd in next_drops)
        if ok:
            k_star = start_k + i + 1
            break

    # fallback: choose arrival k of largest relative drop
    if k_star is None:
        i_argmax = int(np.nanargmax(rel_drop))
        k_star = start_k + i_argmax + 1

    return k_star



