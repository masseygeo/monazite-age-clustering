
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde, t
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle




def plot_samples_panel(df, ax, color_map_col, color_map, label_bool=True, timeline=[210, 450], ylabel='Age (Ma)', kde=True):
    

    ##### plot sample curves...
    sample_max = 0.0                                                # initialize sample_max to get maximum density
    for _, row in df.iterrows():
        mean = row['age']                                           # sample mean
        error = row['error']                                        # sample error (1 sigma)
        symbology = row[color_map_col]                              # symbology attribute
        t = np.linspace(mean - 4 * error, mean + 4 * error, 100)    # age range to create PDF (+/- 4 sigma)
        density = norm.pdf(t, mean, error)                          # create density curve/points
        sample_max = max(sample_max, density.max())                 # update sample_max

        # plot curve (y: age vs. x: density)
        ax.fill_between(density, t, color=color_map.get(symbology), linewidth=0.5, edgecolor='k', alpha=1)
        ax.set_xlim(0, sample_max+0.01)
    

    ##### adjust ax1 styling...
    ax.set_xlim(0, sample_max+0.001)
    ax.set_xticks([])
    ax.set_ylim(timeline[0], timeline[1])
    ax.set_yticks(np.arange(timeline[0], timeline[1]+1, 10))
    ax.set_yticklabels([str(i) if i%20 == 0 else '' for i in np.arange(timeline[0], timeline[1]+1, 10)])
    ax.tick_params(axis='y', which='major', labelleft=label_bool)
    if label_bool:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(None)
    ax.grid(which='major', axis='y', color='gray', alpha=1, ls="--")

    

    ##### plot KDEs on twinx axis (because usually different densities & needs own scale)...
    if kde:
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
        ax1b = ax.twiny()
        ax1b.plot(density_std, t, color='blue', label='Standard KDE')
        ax1b.plot(density_err, t, color='red', label='Error-weighted KDE')

        
        ##### adjust ax1b styling...
        kde_max = max(density_std.max(), density_err.max())     # get maximum KDE density value
        ax1b.set_xlim(0, kde_max+0.001)                         # set x limits of twinx
        ax1b.set_xticks([])                                     # turn axes off of twinx
        ax1b.set_ylim(timeline[0], timeline[1])
        ax1b.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
        ax1b.grid(axis='y', color='gray', alpha=1, ls='--')
        ax1b.set_axis_off()

        return sample_max, kde_max
    
    else:
        return sample_max



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
    ax.set_xlabel('Y2O3 %', fontsize=8, labelpad=-23)
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(axis='y', color='gray', alpha=1, ls='--')




def plot_ageinterp_panel(df, ax, color_label, color_map, bar_width=0.04, gap=0.02):

    ##### pre-proceessing...
    df = df.sort_values('age').copy()
    ages = df['age'].to_numpy()
    errors = df['error'].to_numpy()


    ##### age/SE bars...
    xstart = gap * 2
    for age, err in zip(ages, errors):
        bar = Rectangle(xy=(xstart, age-err), width=bar_width, height=err*2, facecolor=color_map[color_label], edgecolor='k', linewidth=0.2, alpha=1)
        ax.add_patch(bar)
        xstart += bar_width + gap
    

    #### calculate weighted mean age...   
    stats = cluster_stats(ages, errors, two_sigma=False)


    ##### plot weighted mean and CI ribbon...
    if stats['n'] >= 3:
        y0 = stats['mean'] - stats['ci95_halfwidth']
        y1 = stats['mean'] + stats['ci95_halfwidth']
        ax.axhspan(y0, y1, xmin=0, xmax=1, facecolor=color_map[color_label], alpha=0.5, edgecolor='none')
    ax.axhline(y=stats['mean'], xmin=0, xmax=1, color='k', linewidth=1)


    ##### stylize axis...
    ax.text(0.5, 0.985, 'Interpretation', style='italic', transform=ax.transAxes, fontsize=8, ha='center', va='top')
    ax.set_xticks([])
    ax.tick_params(axis='y', which='major', length=0.0001, labelleft=False)
    ax.grid(axis='y', color='gray', alpha=1, ls='--')





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