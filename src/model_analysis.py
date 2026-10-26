"""Descriptive analyses of frozen MoE predictions and observed physiology.

Groups are argmax(g), numbered from zero. Frequencies describe only the supplied
split (possibly artificially balanced), not population clinical risks. Physiology
uses patient medians with equal patient weight; standardized differences are
relative to this split, not clinical normality thresholds. No data are saved.
IDs retain their original types: use the same convention as df_filtered.

Example (after df_filtered, data_out, models_out = pipeline.run()):
    from model_analysis import analyze_cluster_sepsis_rates, analyze_cluster_physiology
    test_ids = data_out[8]
    rates = analyze_cluster_sepsis_rates(
        model, z_test, y_test, test_ids, df_filtered, device=device_dml)
    physiology = analyze_cluster_physiology(
        model, z_test, y_test, test_ids, df_filtered, device=device_dml)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _prepare(model, z, y, patient_ids, df_filtered, threshold, device):
    """Validate a split, align its observed rows, and perform one inference."""
    ids, labels, embeddings = _numpy(patient_ids), _numpy(y), _numpy(z)
    if ids.ndim != 1 or labels.ndim != 1 or embeddings.ndim != 2:
        raise ValueError('Expected patient_ids (N,), y (N,), and z (N, D).')
    n = len(ids)
    if not n or embeddings.shape[1] == 0 or len(labels) != n or len(embeddings) != n:
        raise ValueError('z, y and patient_ids must have equal, nonzero lengths.')
    if pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError('patient_ids must be nonmissing and unique (one window per patient).')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('y must contain binary labels 0/1 only.')
    if not np.isfinite(embeddings).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('z must be finite and threshold must be in [0, 1].')
    required = {'Patient_ID', 'Hour', 'will_have_sepsis'}
    if not df_filtered.columns.is_unique or not required.issubset(df_filtered.columns):
        raise ValueError('df_filtered needs unique columns including Patient_ID, Hour, will_have_sepsis.')
    rows = df_filtered.loc[df_filtered.Patient_ID.isin(ids)].copy()
    missing = pd.Index(ids).difference(rows.Patient_ID.unique())
    if len(missing):
        raise ValueError(f'{len(missing)} supplied IDs absent from df_filtered; check ID types.')
    if rows.Hour.isna().any() or not pd.api.types.is_numeric_dtype(rows.Hour) or not np.isfinite(rows.Hour).all():
        raise ValueError('Selected patients must have finite numeric Hour values.')
    duplicates = int(rows.duplicated(['Patient_ID', 'Hour']).sum())
    if duplicates:
        raise ValueError(f'{duplicates} duplicate (Patient_ID, Hour) rows among selected patients.')
    grouped = rows.groupby('Patient_ID', sort=False)
    if not np.isin(rows.will_have_sepsis, [0, 1]).all() or (grouped.will_have_sepsis.nunique(dropna=False) != 1).any():
        raise ValueError('will_have_sepsis must be binary and constant within each selected patient.')
    if not np.array_equal(grouped.will_have_sepsis.first().reindex(ids).to_numpy(), labels):
        raise ValueError('y disagrees with will_have_sepsis after Patient_ID alignment.')

    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(embeddings, dtype=torch.float32, device=device))
    if not isinstance(output, (tuple, list)) or len(output) != 4:
        raise ValueError('Model must return (p, g, p_k, h).')
    p, g, p_k, h = map(_numpy, output)
    if (p.shape != (n,) or g.ndim != 2 or g.shape[0] != n or g.shape[1] == 0
            or p_k.shape != g.shape or h.ndim != 2 or h.shape[0] != n or h.shape[1] == 0):
        raise ValueError('Invalid model output dimensions: expected (N,), (N,K), (N,K), (N,H).')
    if not all(np.isfinite(a).all() for a in (p, g, p_k, h)):
        raise ValueError('Model outputs contain NaN or infinity.')
    if any(((a < 0) | (a > 1)).any() for a in (p, g, p_k)) or not np.allclose(g.sum(1), 1, atol=1e-5):
        raise ValueError('p, g and p_k must be probabilities; each g row must sum to 1.')
    if not np.allclose(p, (g * p_k).sum(1), atol=1e-5):
        raise ValueError('p is inconsistent with the MoE weighted expert probabilities.')
    assignments = g.argmax(axis=1)
    individual = pd.DataFrame({'position': np.arange(n), 'Patient_ID': ids,
                               'y': labels, 'p': p, 'predicted_positive': p >= threshold,
                               'group': assignments})
    # Only this join determines patient order; z, y and IDs are never sorted.
    rows = individual[['position', 'Patient_ID']].merge(rows, on='Patient_ID', validate='one_to_many')
    rows = rows.sort_values(['position', 'Hour'], kind='stable')
    inference = dict(patient_ids=ids.copy(), p=p, g=g, p_k=p_k, h=h, assignments=assignments)
    return individual, rows, inference


def _colors(k):
    return sns.color_palette('tab10', n_colors=k)


def _finish(figures, show):
    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)  # Still returned, but no notebook auto-display.


def analyze_cluster_sepsis_rates(model, z, y, patient_ids, df_filtered,
                                 threshold=0.5, device='cpu', show=True):
    """Compare observed prevalence, positive prediction rate and mean probability.

    Parameters: model returns (p,g,p_k,h); z (N,D) and binary y (N,) may be
    NumPy arrays or tensors. patient_ids (N,) specifies their shared order.
    df_filtered contains observed windows for these IDs (other IDs are ignored).
    threshold defines positive predictions; device is the inference device.
    show=False suppresses printing/display and returns the unshown figure.

    Returns individual_results, group_stats (including empty groups), global_stats,
    inference and figure. All rates use patients, not hours. These split-specific
    frequencies are not automatically clinical risks in the real population.
    """
    individual, _, inference = _prepare(model, z, y, patient_ids, df_filtered, threshold, device)
    k = inference['g'].shape[1]
    stats = individual.groupby('group').agg(
        n=('y', 'size'), sepsis_count=('y', 'sum'), sepsis_rate=('y', 'mean'),
        predicted_positive_rate=('predicted_positive', 'mean'), mean_probability=('p', 'mean')
    ).reindex(pd.RangeIndex(k, name='group'))
    stats[['n', 'sepsis_count']] = stats[['n', 'sepsis_count']].fillna(0).astype(int)
    global_stats = pd.Series(dict(n=len(individual), sepsis_count=int(individual.y.sum()),
                                 sepsis_rate=individual.y.mean(),
                                 predicted_positive_rate=individual.predicted_positive.mean(),
                                 mean_probability=individual.p.mean()), name='Global')
    metrics = ['sepsis_rate', 'predicted_positive_rate', 'mean_probability']
    labels = ['Proportion réelle de sepsis', 'Proportion prédite positive', 'Probabilité prédite moyenne']
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(max(9, k * 1.5), 6))
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(np.arange(k) + (j - 1) * .25, stats[metric], width=.25,
               color=_colors(k), hatch=['', '//', '..'][j], label=label)
        ax.axhline(global_stats[metric], color=['black', 'dimgray', 'slategray'][j],
                   linestyle=['-', '--', ':'][j],
                   label=f'Global — {label} : {global_stats[metric]:.1%}')
    ax.set_xticks(np.arange(k), [f'G{i}\nn={stats.loc[i, "n"]}' for i in range(k)])
    ax.set(ylim=(0, 1.05), ylabel='Pourcentage', title=f'Sepsis par groupe — seuil prédictif {threshold:g}')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    if show:
        print(pd.concat([stats, global_stats.to_frame().T]).to_string())
    _finish([fig], show)
    return dict(individual_results=individual, group_stats=stats,
                global_stats=global_stats, inference=inference, figure=fig)


def _physio_stats(values, variables):
    """One row per variable; denominators include patients without measurements."""
    return pd.DataFrame({
        'median': values.median(), 'q1': values.quantile(.25), 'q3': values.quantile(.75),
        'n_measured': values.count(),
        'coverage_pct': values.count() * 100 / len(values) if len(values) else np.nan,
    }).reindex(variables).rename_axis('variable')


def _heatmap(table, title, comparative=False):
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(max(7, table.shape[1] * 1.2), max(3, len(table) * .4)))
    options = dict(vmin=0, vmax=100, cmap='Blues', cbar_kws={'label': 'Patients mesurés (%)'})
    if comparative:
        finite = table.to_numpy()[np.isfinite(table.to_numpy())]
        limit = max(float(np.abs(finite).max()), .01) if finite.size else 1
        options = dict(vmin=-limit, vmax=limit, center=0, cmap='vlag',
                       cbar_kws={'label': '(Médiane groupe − médiane globale) / IQR global'})
    sns.heatmap(table, mask=table.isna(), annot=True, fmt='.1f', ax=ax, **options)
    for row, col in zip(*np.where(table.isna().to_numpy())):
        ax.text(col + .5, row + .5, 'ND', ha='center', va='center', fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def analyze_cluster_physiology(model, z, y, patient_ids, df_filtered,
                              threshold=0.5, device='cpu', show=True, physio_variables=None):
    """Summarize observed physiology using equally weighted patient medians.

    Common parameters follow analyze_cluster_sepsis_rates. physio_variables is
    an optional list of numeric physiological columns; default excludes technical,
    label, demographic and exported-index columns. Values retain original units
    and NaNs. Each patient summary includes median, observed count and window hours.
    Group/global quartiles are computed directly across patient medians.

    Returns aligned individual_results and patient_summaries (long format),
    group_stats, global_stats, coverage (%), summary_table, relative_differences,
    inference, and figures (comparison, coverage, boxplots). Undefined statistics
    remain NaN, including relative differences when global IQR is zero/missing.
    show=False suppresses printing/display. No imputation, fitting or saving.
    """
    individual, rows, inference = _prepare(model, z, y, patient_ids, df_filtered, threshold, device)
    excluded = {'Patient_ID', 'Hour', 'will_have_sepsis', 'SepsisLabel', 'sepsis_onset_hour',
                'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
                'index', 'level_0', 'position'}
    excluded.update(c for c in df_filtered if str(c).startswith('Unnamed:'))
    variables = list(physio_variables) if physio_variables is not None else [
        c for c in df_filtered.select_dtypes(include=np.number) if c not in excluded]
    if not variables or len(set(variables)) != len(variables):
        raise ValueError('Provide at least one unique physiological variable.')
    if any(c not in df_filtered or c in excluded or not pd.api.types.is_numeric_dtype(df_filtered[c])
           for c in variables):
        raise ValueError('Physiological variables must exist, be numeric, and exclude technical columns/labels.')
    if np.isinf(rows[variables].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError('Observed physiology contains infinite values.')
    ids = individual.Patient_ID.to_numpy()
    grouped = rows.groupby('Patient_ID', sort=False)
    medians = grouped[variables].median().reindex(ids)
    counts = grouped[variables].count().reindex(ids)
    hours = grouped.Hour.nunique().reindex(ids)
    summaries = pd.concat([
        individual.assign(variable=v, median=medians[v].to_numpy(),
                          n_observed=counts[v].to_numpy(), n_hours=hours.to_numpy())
        for v in variables
    ], ignore_index=True).sort_values('position', kind='stable').reset_index(drop=True)
    k = inference['g'].shape[1]
    stats = pd.concat([_physio_stats(medians.iloc[np.flatnonzero(individual.group.to_numpy() == i)], variables)
                       for i in range(k)], keys=range(k), names=['group'])
    global_stats = _physio_stats(medians, variables)
    coverage = stats.coverage_pct.unstack('group').reindex(variables)
    coverage.columns = [f'G{i}' for i in range(k)]
    coverage['Global'] = global_stats.coverage_pct
    iqr = global_stats.q3 - global_stats.q1
    differences = stats['median'].unstack('group').reindex(variables).sub(global_stats['median'], axis=0)
    differences = differences.div(iqr.where(iqr > 0), axis=0)
    differences.columns = [f'G{i}' for i in range(k)]
    combined = pd.concat({**{f'G{i}': stats.loc[i] for i in range(k)}, 'Global': global_stats})
    formatted = combined.apply(lambda r: 'ND' if pd.isna(r['median']) else
                               f'{r["median"]:.3g} [{r["q1"]:.3g}–{r["q3"]:.3g}]', axis=1)
    table = formatted.unstack(0).reindex(index=variables, columns=list(coverage.columns))
    figures = {
        'comparison': _heatmap(differences, 'Écarts relatifs au jeu analysé — pas des seuils cliniques\nND : IQR nul/manquant ou groupe non mesuré', True),
        'coverage': _heatmap(coverage, 'Couverture : patients avec au moins une mesure'),
        'boxplots': [],
    }
    colors = _colors(k)
    for start in range(0, len(variables), 6):
        panel = variables[start:start + 6]
        with plt.ioff():
            fig, axes = plt.subplots((len(panel) + 2) // 3, 3, figsize=(15, 4 * ((len(panel) + 2) // 3)), squeeze=False)
        for ax, variable in zip(axes.flat, panel):
            for i in range(k):
                values = medians.loc[individual.group.to_numpy() == i, variable].dropna()
                if len(values):
                    box = ax.boxplot([values.to_numpy()], positions=[i], widths=.6, patch_artist=True)
                    box['boxes'][0].set_facecolor(colors[i])
            reference = global_stats.loc[variable, 'median']
            if pd.notna(reference):
                ax.axhline(reference, color='black', linestyle='--', label='Médiane globale')
                ax.legend(fontsize=8)
            ax.set_xticks(range(k), [f'G{i}' for i in range(k)])
            ax.set(xlim=(-.6, k - .4), title=variable, ylabel='Médiane individuelle (unité originale)')
        for ax in list(axes.flat)[len(panel):]:
            ax.set_visible(False)
        fig.tight_layout()
        figures['boxplots'].append(fig)
    if show:
        print('Médiane [Q1–Q3] des médianes individuelles :\n', table.to_string())
        print('\nCouverture des patients (%) :\n', coverage.to_string())
    _finish([figures['comparison'], figures['coverage'], *figures['boxplots']], show)
    return dict(individual_results=individual, patient_summaries=summaries,
                group_stats=stats, global_stats=global_stats, coverage=coverage,
                summary_table=table, relative_differences=differences,
                inference=inference, figures=figures)
