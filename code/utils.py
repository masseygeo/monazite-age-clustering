
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklearn.cluster import KMeans



texture_colors = {'core': '#239BA7', 
                  'overgrowth': '#ECECBB', 
                  'rim': '#E1AA36'}

# texture_colors = {'core': '#9CAFAA', 
#                   'overgrowth': '#D6DAC8', 
#                   'rim': '#FBF3D5'}


age_group_colors = {0:'#f25117', 
                    1:'#f2a104', 
                    2:'#00743f', 
                    3:'#72a2c0', 
                    4:'#1d65a6', 
                    5:'#192e5b',
                    6:'k',
                    7:'red',
                    8:'blue',
                    9:'darkred',
                    10:'green',
                    -1:'#b1b8bc'}

gap = 0.3
bar_width = 0.4
timeline=[210, 450]


def plot_samples_panel(df, ax, color_map_col, color_map, label_bool=True, timeline=[210, 450], ylabel='Age (Ma)'):
    

    # initialize sample_max to get maximum density
    sample_max = 0.0
    
    # iterate through samples...
    for _, row in df.iterrows():

        # get data...
        mean = row['age']                                           # sample mean
        error = row['error']                                        # sample error (1 sigma)
        symbology = row[color_map_col]                              # symbology attribute
        t = np.linspace(mean - 4 * error, mean + 4 * error, 100)    # age range to create PDF (+/- 4 sigma)
        density = norm.pdf(t, mean, error)                          # create density curve/points
        sample_max = max(sample_max, density.max())                 # update sample_max

        # plot curve (y: age vs. x: density)...
        ax.fill_between(density, t, color=color_map.get(symbology), linewidth=0.5, edgecolor='k', alpha=1)
        ax.set_xlim(0, sample_max+0.01)
    
    # adjust ax styling...
    ax.set_xlim(0, sample_max+0.001)
    ax.set_xticks([])
    ax.set_ylim(timeline[0], timeline[1])
    ax.set_ylim(ax.get_ylim())
    ax.set_yticks(np.arange(timeline[0], timeline[1]+1, 10))
    ax.set_yticklabels([str(i) if i%20 == 0 else '' for i in np.arange(timeline[0], timeline[1]+1, 10)])
    ax.tick_params(axis='y', which='major', labelleft=label_bool)
    if label_bool:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(None)
    ax.grid(which='major', axis='y', color='gray', alpha=1, ls="--")

    # ax.set_title('Density')

    return sample_max

    


def plot_kde_panel(df, ax, timeline=[210, 450]):

    # get basic info from df...
    ages = df['age'].to_numpy()                        # array of means
    errors = df['error'].to_numpy()                    # array of errors (1 sigma)
    t = np.linspace(timeline[0], timeline[-1], 1000)   # array of age range values

    # calculate standard KDE...
    kde = gaussian_kde(ages)
    density_std = kde(t)

    # calculate error-weighted KDE...
    density_err = np.zeros_like(t)
    for age, err in zip(ages, errors):
        density_err += np.exp(-(t-age)**2 / (2*err**2)) / (np.sqrt(2*np.pi)*err)
    density_err /= len(ages)

    # plot KDEs on twiny axis...
    ax.plot(density_std, t, color='blue', label='Standard KDE')
    ax.plot(density_err, t, color='red', label='Error-weighted KDE')
    
    # adjust ax styling...
    kde_max = max(density_std.max(), density_err.max())     # get maximum KDE density value
    ax.set_xlim(0, kde_max+0.001)                           # set x limits of twinx
    ax.set_xticks([])                                       # turn axes off
    # ax.set_ylim(timeline[0], timeline[1])

    

    ax.set_ylim(ax.get_ylim())
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(axis='y', color='gray', alpha=1, ls='--')
    ax.set_axis_off()

    return kde_max



def plot_yttrium_panel(df, ax, y_col, y_symbology):

    ##### plot Age vs. Y2O3 (symbolized by garnet presence)
    grt = df[df[y_symbology] == 1].copy()
    no_grt = df[df[y_symbology] == 0].copy()

    ax.plot(grt[y_col], grt['age'], marker='o', markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.3, linewidth=0, alpha=0.5)
    ax.plot(no_grt[y_col], no_grt['age'], marker='o', markerfacecolor='blue', markeredgecolor='k', markeredgewidth=0.3, linewidth=0, alpha=0.5)


    ##### adjust ax styling...
    ax.set_xlim(df[y_col].min() - 0.3, df[y_col].max() + 0.3)
    ax.set_xticks([df[y_col].min(), df[y_col].max()])
    ax.set_xticklabels([round(df[y_col].min(), 2), round(df[y_col].max(), 2)], fontsize=8)
    ax.set_xlabel('Y2O3 (%)', fontsize=8, labelpad=-23)
    # ax.set_title('Y2O3 (Wt. %)')
    ax.set_ylim(ax.get_ylim())
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(axis='y', color='gray', alpha=1, ls='--')




def plot_ageinterp_panel(df, ax, color_label, color_map, bar_width=0.04, gap=0.02):

    ##### pre-proceessing...
    df = df.sort_values('age').copy()
    ages = df['age'].to_numpy()
    errors = df['error'].to_numpy()


    #### calculate weighted mean age...   
    stats = cluster_stats(ages, errors, two_sigma=False)


    ##### plot weighted mean CI ribbon (on bottom)...
    if stats['n'] >= 3:
        y0 = stats['mean'] - stats['ci95_halfwidth']
        y1 = stats['mean'] + stats['ci95_halfwidth']
        ax.axhspan(y0, y1, xmin=0, xmax=1, facecolor=color_map[color_label], alpha=0.35, edgecolor='none')


    ##### plot age/SE bars for each group...
    xstart = gap * 2
    for age, err in zip(ages, errors):
        bar = Rectangle(xy=(xstart, age-err), width=bar_width, height=err*2, facecolor=color_map[color_label], edgecolor='k', linewidth=0.4)
        ax.add_patch(bar)
        xstart += bar_width + gap
    

    ##### plot weighted mean line...
    if stats['n'] >= 3:
    #     y0 = stats['mean'] - stats['ci95_halfwidth']
    #     y1 = stats['mean'] + stats['ci95_halfwidth']
    #     ax.axhspan(y0, y1, xmin=0, xmax=1, facecolor=color_map[color_label], alpha=0.5, edgecolor='none')
        ax.axhline(y=stats['mean'], xmin=0, xmax=1, color='k', linewidth=1)


    ##### stylize axis...
    ax.text(0.5, 0.985, 'Interpretation', style='italic', transform=ax.transAxes, fontsize=8, ha='center', va='top')

    # ax.set_title('Weighted Mean Ages')

    ax.set_xticks([])
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(axis='y', color='gray', alpha=1, ls='--')
    ax.set_ylim(ax.get_ylim())

    if stats['n'] > 3:
        return stats




def cluster_stats(ages, se, two_sigma=True):

    n    = len(ages)
    w    = 1.0 / (se**2)

    # Weighted mean
    mu = np.sum(w * ages) / np.sum(w)

    # MSWD (χ²/(n-1))
    chi2 = np.sum(w * (ages - mu)**2)
    mswd = chi2 / max(n - 1, 1)

    # Uncertainty on the mean (inflate if MSWD>1)
    sigma_mu = np.sqrt(1.0 / np.sum(w))
    sigma_mu_adj = sigma_mu * np.sqrt(mswd) if mswd > 1 else sigma_mu

    # 95% CI (or 2σ if you prefer)
    if two_sigma:
        half_width = 2.0 * sigma_mu_adj
    else:
        tcrit = t.ppf(0.975, df=max(n - 1, 1))
        half_width = tcrit * sigma_mu_adj

    return {
        "mean": mu,
        "mswd": mswd,
        "n": n,
        "sigma_mean": sigma_mu_adj,
        "ci95_halfwidth": half_width
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