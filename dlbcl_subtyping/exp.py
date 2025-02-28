import os
import warnings
from glob import glob
from tqdm import tqdm
from pydantic import Field
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seahorse as sns
import h5py
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan
import torch
import timm

from .utils.cli import BaseMLCLI, BaseMLArgs

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass

    class ClusterArgs(CommonArgs):
        target: str = Field('cluster', s='-T')
        noshow: bool = False

    def run_cluster(self, a):
        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
        with h5py.File('data/slide_features.h5', 'r') as f:
            features = f['features'][:]
            df = pd.DataFrame({
                'name': [int((v.decode('utf-8'))) for v in f['names'][:]],
                'filename': [v.decode('utf-8') for v in f['filenames'][:]],
                'order': f['orders'][:],
            })

        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        print('Loaded features', features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        # scaled_features = features

        print('UMAP fitting...')
        reducer = umap.UMAP(
                n_neighbors=10,
                min_dist=0.05,
                n_components=2,
                metric='cosine',
                random_state=a.seed,
                n_jobs=1,
            )
        embedding = reducer.fit_transform(scaled_features)
        print('Loaded features', features.shape)

        if a.target in [
                'HDBSCAN',
                'CD10 IHC', 'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH',
                'ECOG PS', 'LDH', 'EN', 'Stage', 'IPI Score',
                'IPI Risk Group (4 Class)', 'RIPI Risk Group', 'Follow-up Status',
                ]:
            mode = 'categorical'
        elif a.target in ['MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'Age', 'OS', 'PFS']:
            mode = 'numeric'
        else:
            raise RuntimeError('invalid target', a.target)

        marker_size = 15
        if mode == 'categorical':
            if a.target == 'cluster':
                eps = 0.2
                m = hdbscan.HDBSCAN(
                    min_cluster_size=5,
                    min_samples=5,
                    cluster_selection_epsilon=eps,
                    metric='euclidean',
                )
                labels = m.fit_predict(embedding)
                n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            else:
                labels = df[a.target].fillna(-1)
                n_labels = len(set(labels))

            plt.figure(figsize=(10, 8))
            cmap = plt.cm.viridis

            noise_mask = labels == -1
            valid_labels = sorted(list(set(labels[~noise_mask])))
            norm = plt.Normalize(min(valid_labels or [0]), max(valid_labels or [1]))
            for label in valid_labels:
                mask = labels == label
                color = cmap(norm(label))
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1], c=[color],
                    s=marker_size, label=f'{a.target} {label}'
                )

            if np.any(noise_mask):
                plt.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray',
                    s=marker_size, marker='x', label='Noise/NaN',
                )

        else:
            values = df[a.target]
            norm = Normalize(vmin=values.min(), vmax=values.max())
            values = values.fillna(-1)
            has_value = values > 0
            cmap = plt.cm.viridis
            scatter = plt.scatter(embedding[has_value, 0], embedding[has_value, 1], c=values[has_value],
                                  s=marker_size, cmap=cmap, norm=norm, label=a.target,)
            if np.any(has_value):
                plt.scatter(embedding[~has_value, 0], embedding[~has_value, 1], c='gray',
                            s=marker_size, marker='x', label='NaN')
            cbar = plt.colorbar(scatter)
            cbar.set_label(a.target)

        plt.title(f'UMAP + {a.target}')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend()
        plt.tight_layout()
        os.makedirs('out/umap', exist_ok=True)
        name = a.target.replace(' ', '_')
        plt.savefig(f'out/umap/umap_{name}.png')

        if not a.noshow:
            plt.show()

if __name__ == '__main__':
    cli = CLI()
    cli.run()
