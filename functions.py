from vars import *
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def get_data(file_name, columns):
    data_all = pd.DataFrame(columns=columns)

    f = open(file_name, 'r', encoding="utf8")
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)['teamTableStats']
        data = pd.DataFrame(data)[columns]
        data_all = pd.concat([data_all, data], ignore_index=True)

    return data_all


def standardize(df, op):
    op = columns_in_options[op][0]
    return df[op].apply(
        lambda x: (x - df[op].mean()) / (df[op].var() ** (1 / 2))
    )


def normalize(df, op):
    op = columns_in_options[op][0]
    return df[op].apply(
        lambda x: (x - df[op].min()) / (df[op].max() - df[op].min())
    )


def write_describe_to_excel(df):
    desc = df[options].describe(include='all').T
    desc['коэффициент вариации'] = desc['std'] / desc['mean']
    desc['нижняя граница выбросов'] = desc['mean'] - 3 * desc['std']
    desc['верхняя граница выбросов'] = desc['mean'] + 3 * desc['std']
    desc.to_excel('xlsx/describe.xlsx')


def write_corr_pirs_to_excel(df):
    df[options].corr().to_excel('xlsx/corr_pirs.xlsx')


def cluster(df, k, name):
    fig, ax = plt.subplots(1, 1)

    kmeans = KMeans(n_clusters=k, n_init=50).fit(df[options])
    df['cluster'] = kmeans.labels_
    centers = kmeans.cluster_centers_
    n = 1
    for center in centers:
        ax.plot(range(len(options)), center, marker='o', label=str(n) + ' кластер')
        n += 1

    ax.set_xticklabels([''] + list(norm_names_options.values()))
    ax.set_ylabel('значение центра')
    ax.legend()

    fig.savefig(f"plots/clusters/{name}.png")
    plt.close(fig)

    df.to_excel(f'xlsx/clust_{name}.xlsx')


def plot_scatter(ops, df):
    fig, ax = plt.subplots(figsize=(15, 10))

    x, y = df[ops[0]], df[ops[1]]

    ax.scatter(x=x, y=y, color="navy")
    plt.xlabel(norm_names_options[ops[0]], fontsize=25)
    plt.ylabel(norm_names_options[ops[1]], fontsize=25)

    fig.savefig(f"plots/{ops[0] + '_' + ops[1]}.png")  # save the figure to file
    print('saved')
    plt.close(fig)


def plot_dend(df, name, method='ward'):
    plt.figure(figsize=(15, 10))
    Z = linkage(df[options], method=method)
    dendrogram(Z, no_labels=True, link_color_func=lambda k: 'black')

    plt.xlabel('клубы', fontsize=25)
    plt.ylabel('расстояние', fontsize=25)

    plt.savefig(f'plots/dend/{name}_{method}.png')


def plot_elbow_method(df):
    s, e = 2, 13
    crit = []
    for k in range(s, e):
        kmeans = KMeans(n_clusters=k, n_init=50).fit(df['inital'][options])
        crit.append(kmeans.inertia_)

    plt.figure(figsize=(15, 10))
    plt.plot(range(s, e), crit, color='navy')
    plt.xlabel('количество кластеров', fontsize=25)
    plt.ylabel('сумма квадратов расстояний до центров', fontsize=25)

    plt.savefig(f'plots/elbow_2.png')


def plot_avg_silhouette(df):
    s, e = 2, 13
    sil = []
    for k in range(s, e):
        kmeans = KMeans(n_clusters=k, n_init=50).fit(df['inital'][options])
        sil.append(silhouette_score(df['inital'][options], kmeans.labels_))

    plt.figure(figsize=(15, 10))
    plt.plot(range(s, e), sil, color='navy')
    plt.xlabel('количество кластеров', fontsize=25)
    plt.ylabel('силуэт', fontsize=25)

    plt.savefig(f'plots/sil.png')


def plot_silhouette(df):
    X = df['inital'][options]

    for n_clusters in range(4, 9):
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_xlabel("силуэт", fontsize=25)
        ax1.set_ylabel("кластер", fontsize=25)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks

        fig.savefig(f"plots/sil/n_{n_clusters}.png")