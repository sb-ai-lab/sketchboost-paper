import os
import joblib
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from pandas import Series, DataFrame


def detailed_plot(dataset_order, strategy_order, attr, st_d, nm_d, alias=None, legend=True, baselines=None, k_limit=20,
                  output_dir='OUTPUT_MAIN'):
    path = os.path.join(output_dir, 'detailed_' + attr)
    os.makedirs(path, exist_ok=True)

    if alias is None:
        alias = attr

    for ds in dataset_order:

        plt.figure(figsize=(7, 5))
        for strategy in strategy_order:
            tt = joblib.load(os.path.join('agg', ds, strategy + '.pkl'))[attr]

            tt = tt[tt.index <= k_limit]
            tt.name = st_d[strategy]
            tt.plot(title=ds, marker='o')

        stat = joblib.load(os.path.join('agg', ds, 'baselines.pkl'))[attr]
        if baselines is not None:
            stat = stat[[x for x in baselines if x in stat.index]]

        for bs in stat.index:
            sc = stat[bs]
            Series([sc] * 2, index=[tt.dropna().index.min(), tt.dropna().index.max()], name=st_d[bs]).plot(
                linestyle='--', lw=2)

        plt.title(nm_d[ds], fontsize=25)
        if legend:
            plt.legend(loc=10, bbox_to_anchor=(0.5, -0.25, 0, 0), ncol=3, fontsize=12)
        plt.xlabel('Reduced Dimension', fontsize=15)
        plt.ylabel(alias, fontsize=15)
        plt.xticks(np.arange(max(tt.dropna().index)) + 1, np.arange(max(tt.dropna().index)) + 1)
        plt.savefig(os.path.join(path, ds + '.png'), dpi=720)


def summary_plot(dataset_order, strategy_order, attr, alias, st_d, nm_d, r=0, baselines=None, sorter='test_score',
                 k_limit=20,
                 output_dir='OUTPUT_MAIN'):
    path = os.path.join(output_dir, 'summary_' + attr)
    os.makedirs(path, exist_ok=True)

    if alias is None:
        alias = attr

    for ds in dataset_order:

        plt.figure(figsize=(8.5, 6))
        best_scores_times = {}

        for strategy in strategy_order:
            stat = joblib.load(os.path.join('agg', ds, strategy + '.pkl')).sort_values(sorter)
            stat = stat[stat.index <= k_limit].iloc[0]

            tt = stat[attr]
            best_scores_times[strategy] = tt

        stat = joblib.load(os.path.join('agg', ds, 'baselines.pkl'))[attr]
        if baselines is not None:
            stat = stat[[x for x in baselines if x in stat.index]]

        for bs in stat.index:
            best_scores_times[bs] = stat[bs]

        best_scores_times = Series(best_scores_times, name=alias)
        best_scores_times.index.name = 'Strategy'
        best_scores_times = best_scores_times.reset_index()
        best_scores_times['Strategy'] = best_scores_times['Strategy'].map(st_d)
        rects = sns.barplot('Strategy', alias,
                            data=best_scores_times, edgecolor="#034569")

        max_ = best_scores_times[alias].max()

        for index, row in best_scores_times.iterrows():
            val = int(row[alias]) if r == 0 else round(row[alias], r)
            rects.text(index, row[alias] + max_ * 0.02, val, color='black', ha="center")

        plt.ylim(0, max_ * 1.08)
        plt.xticks(rotation=12)
        plt.xlabel('Strategy', fontsize=15)
        plt.ylabel(alias, fontsize=15)

        plt.title(nm_d[ds], fontsize=25)

        plt.savefig(os.path.join(path, ds + '.png'), dpi=720)


def get_summary_table(dataset_order, strategy_order, attr, st_d, nm_d, r=4, baselines=None, sorter='test_score',
                      k_limit=20,
                      output_dir='OUTPUT_DIR'):
    res = {}

    for ds in dataset_order:

        best_scores = {}
        stat = joblib.load(os.path.join('agg', ds, 'baselines.pkl'))[attr]
        if baselines is not None:
            stat = stat[[x for x in baselines if x in stat.index]]

        for bs in stat.index:
            best_scores[st_d[bs]] = stat[bs]

        for strategy in strategy_order:
            stat = joblib.load(os.path.join('agg', ds, strategy + '.pkl')).sort_values(sorter)
            stat = stat[stat.index <= k_limit].iloc[0]
            best_scores[st_d[strategy]] = stat[attr]

        res[ds] = best_scores

    res[ds] = best_scores

    res = DataFrame(res).T

    if r == 0:
        res = res.fillna(0).astype(int)
        res[res == 0] = '-'
    else:
        res = res.round(r)
        res[res.isnull()] = '-'

    res.index.name = 'Dataset'
    res = res.reset_index()
    res['Dataset'] = res['Dataset'].map(nm_d)

    res.to_csv(os.path.join(output_dir, 'summary_' + attr + '.csv'), index=False)

    return res


def get_total_table(dataset_order, strategy_order, attr, st_d, nm_d, r=4, baselines=None, k_limit=20,
                    output_dir='OUTPUT_MAIN'):
    res = {}
    un_bs = set()
    for ds in dataset_order:

        best_scores = {}
        stat = joblib.load(os.path.join('agg', ds, 'baselines.pkl'))[attr]
        if baselines is not None:
            stat = stat[[x for x in baselines if x in stat.index]]

        for bs in stat.index:
            best_scores[(st_d[bs], '-')] = stat[bs]
            un_bs.add(bs)

        for strategy in strategy_order:

            tt = joblib.load(os.path.join('agg', ds, strategy + '.pkl'))[attr]

            for i in tt.index.values:
                best_scores[(st_d[strategy], int(i))] = tt[i]

        res[ds] = best_scores

    res[ds] = best_scores

    res = DataFrame(res).loc[[st_d[x] for x in list(un_bs) + strategy_order]]

    if r == 0:
        res = res.fillna(0).astype(int)
        res[res == 0] = '-'
    else:
        res = res.round(r)
        res[res.isnull()] = '-'

    res.columns = [nm_d[x] for x in res.columns]
    res.index.names = ('Dataset', 'K')
    res = res.reset_index()

    res.to_csv(os.path.join(output_dir, 'detailed_' + attr + '.csv'), index=False)

    return res
