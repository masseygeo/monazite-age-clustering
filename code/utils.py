
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN




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

# settings to determine
gap = 0.3
bar_width = 0.4
timeline=[210, 450]


def plot_samples_panel(df, ax, color_col, color_map, label_bool=True, timeline=timeline, y_label='Age (Ma)'):
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
        timeline (list[int], optional): Lower and upper bounds of the age axis (Ma).
            Defaults to [210, 450].
        ylabel (str, optional): Y-axis label text. Defaults to "Age (Ma)".

    Returns:
        float: Maximum density value across all plotted samples, useful for
        scaling ax.
    """
    
    # initialize sample_max to get maximum density
    sample_max = 0.0
    
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
        ax.fill_between(density, t, color=color_map.get(symbology), linewidth=0.5, edgecolor='k', alpha=1)
        ax.set_xlim(0, sample_max+0.01)
    
    # adjust ax styling...
    ax.set_xlim(0, sample_max+0.001)                                    # set x limits according to density max
    ax.set_xticks([])                                                   # turn off x ticks
    # ax.set_ylim(timeline[0], timeline[1])
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

    


def plot_kde_panel(df, ax, timeline=timeline):
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
        timeline (list[int], optional): Lower and upper bounds of the age axis (Ma).
            Defaults to [210, 450].

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



# gap = 0.3
# bar_width = 0.4
def plot_ageinterp_panel(df, ax, color_col, color_map, bar_width=bar_width, gap=gap, ci='2sigma'):
    """
    Plot weighted mean age metrics (individual ages, errors, and MSWD).

    The function:
      1. Computes the weighted mean age and confidence interval for the group.
      2. If at least 3 ages are present, plots a semi-transparent 95% CI ribbon
         around the mean at the bottom of the axis.
      3. Plots vertical bars for each individual sample age with +/-1 sigma error.
      4. If at least 3 ages are present, overlays a horizontal line at the
         weighted mean age.

    Args:
        df (pandas.DataFrame): Input DataFrame containing:
            - 'age' (float): Sample ages.
            - 'error' (float): Sample age errors (1 sigma SE).
        ax (matplotlib.axes.Axes): Axis object to plot on.
        color_col (str): Name of the category (key in `color_map`) used to
            determine plotting color.
        color_map (dict): Mapping of categories to matplotlib colors.
        bar_width (float, optional): Width of the individual age bars.
            Defaults to global `bar_width`.
        gap (float, optional): Horizontal gap between bars. Defaults to global `gap`.
        ci (str, optional): Confidence interval to use for weighted mean ('2sigma' or '95%'). Defaults to '2sigma'.

    Returns:
        dict or None: Weighted mean statistics dictionary (from `cluster_stats`) if
        the group has more than 3 ages; otherwise, returns None.
    """

    # pre-proceessing...
    df = df.sort_values('age').copy()   # sort df by age (ascending)
    ages = df['age'].to_numpy()         # extract age array
    errors = df['error'].to_numpy()     # extract error array (1 sigma SE)

    # calculate weighted mean age and stats...   
    stats = cluster_stats(ages, errors, ci='2sigma')

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

    if stats['n'] > 3:
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




def horizontal_stack(ax, k_label, yval, counts_dict, count_color_dict, violin_arr, violin_max):

    x0 = 0
    total = sum(counts_dict.values())

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([])
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(False)

    ax.text(x=0, y=yval, s=k_label, fontsize=7, weight='bold')

    for tex, count in counts_dict.items():
        x1 = count / total
        count_bar = Rectangle(
            xy=(x0, yval+12), width=x1, height=-10,
            facecolor=count_color_dict[tex], linewidth=0.5, edgecolor='k'
        )
        ax.add_patch(count_bar)
        x0 += x1




    if not hasattr(ax, "_box_overlay"):
        box_ax = ax.figure.add_axes(ax.get_position(), frameon=False)  # new axes, no sharing
        box_ax.set_yticks([]); box_ax.set_xticks([])
        box_ax.patch.set_alpha(0.0)
        box_ax.set_zorder(ax.get_zorder() + 1)
        ax._box_overlay = box_ax
    else:
        box_ax = ax._box_overlay
        # keep overlay aligned with parent in case layout updated
        box_ax.set_position(ax.get_position())

    box_ax.set_axis_off()

    # keep overlay in sync with parent limits but NOT shared
    box_ax.set_ylim(ax.get_ylim())
    # box_ax.set_yticks([])
    # box_ax.set_xticks([])
    box_ax.set_xlim(-0.1, violin_max+0.2)

    # (optional) clamp to stay within y-lims so it never nudges bounds visually
    ylo, yhi = box_ax.get_ylim()
    pos = max(min(yval + 19, max(ylo, yhi) - 1), min(ylo, yhi) + 1)

    # draw the boxplot on the overlay axis
    box_ax.boxplot(
        violin_arr, positions=[pos], vert=False, widths=10, patch_artist=True,
        boxprops=dict(facecolor='tab:blue', color='k', linewidth=0.5),
        medianprops=dict(color='k', linewidth=0.5),
        whiskerprops=dict(color='k', linewidth=0.5),
        capprops=dict(color='k', linewidth=0.5),
        flierprops=dict(color='k', markersize=2, linewidth=0.25, markerfacecolor='k')
    )
    




def kmeans_processing(df, k, weights=False):

    ages = df['age'].to_numpy().reshape(-1,1)
    if not weights:
        kmeans = KMeans(n_clusters=k, random_state=111, n_init='auto').fit(ages) 
    else:
        errors = df['error'].to_numpy().reshape(-1,1)
        weights = 1/errors**2
        kmeans = KMeans(n_clusters=k, random_state=111, n_init='auto').fit(ages, sample_weight=weights) 

    # sort cluster centers numerically...
    order = np.argsort(kmeans.cluster_centers_.ravel())
    labels = kmeans.labels_
    relabel = np.empty_like(order)
    relabel[order] = np.arange(order.size)
    df['labels'] = relabel[labels]
    kmeans.cluster_centers_ = kmeans.cluster_centers_[order]

    # sort the df according to labels & error...
    df.sort_values(['labels', 'error'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df



def gmm_processing(df, k, use_weights=False, age_min=210, age_max=450, n_pts=1000):
    ages = df['age'].to_numpy().reshape(-1, 1)

    if use_weights:
        errors = df['error'].to_numpy().reshape(-1, 1)
        sample_wts = (1.0 / (errors**2)).ravel()
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, random_state=111).fit(ages, sample_weight=sample_wts)
    else:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, random_state=111).fit(ages)

    # Extract params
    means = gmm.means_.ravel()                        # (k,)
    stds  = np.sqrt(gmm.covariances_.reshape(-1)).ravel()  # (k,)
    weights = gmm.weights_.ravel()                    # (k,)

    # Order by mean (young→old)
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    # Build timeline for plotting
    timeline = np.linspace(age_min, age_max, n_pts)

    # Component PDFs (weighted)
    stds_plot = np.clip(stds, 0.5, None)  # plotting-only guard
    components = [w * norm.pdf(timeline, loc=mu, scale=sd)
                  for mu, sd, w in zip(means, stds_plot, weights)]

    # Responsibilities & labels mapped to ordered components
    resp = gmm.predict_proba(ages)[:, order]
    labels = resp.argmax(axis=1)

    df = df.copy()
    df['labels'] = labels
    df.sort_values(['labels', 'error'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Optional: return total mixture curve for convenience
    mixture = np.sum(components, axis=0)

    return df, components, timeline, mixture




def plot_components_panel(ax, components, timeline, mixture, color_dict):

    for idx, c in enumerate(components):
        # ax.plot(c, timeline, color=color_dict[idx], linewidth=2.0, linestyle='--')
        ax.plot(c, timeline, color='k', linewidth=1.5, linestyle='--')

    ax.set_xlim(0, np.max(components))
    ax.set_xticks([])
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(False)
    ax.set_axis_off()




def dbscan_processing(df, eps, min_samples):
    ages = df['age'].to_numpy().reshape(-1, 1)

    dbs = DBSCAN(eps=eps, min_samples=min_samples).fit(ages)
    df = df.copy()  # avoid mutating caller's df
    df['labels'] = dbs.labels_

    # 1) Mean age per *cluster* (exclude noise = -1)
    means = (df.loc[df['labels'] != -1].groupby('labels')['age'].mean().sort_values())

    # 3) Build mapping old -> new (0..K-1) in ascending mean age
    remap = {old_label: new_idx for new_idx, old_label in enumerate(means.index)}
    K = len(remap)

    # Apply mapping; keep noise as -1
    df['labels'] = df['labels'].map(remap).fillna(-1).astype(int)

    # (Optional) group mean (after relabel) if you still want it
    # df['mean age'] = df.groupby('labels')['age'].transform('mean')

    # 4) Sort with noise (-1) last, then by error (or age if you prefer)
    sort_key = np.where(df['labels'] == -1, K, df['labels'])
    df = df.iloc[np.lexsort((df.get('error', 0), sort_key))].reset_index(drop=True)

    df = df.sort_values(['labels', 'error'])

    return df