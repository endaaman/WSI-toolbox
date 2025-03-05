
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
from sklearn.decomposition import PCA
import hdbscan
import torch
import timm

from .utils import BaseMLCLI, BaseMLArgs

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass



    class ExtractGlobalFeaturesArgs(CommonArgs):
        noshow: bool = False

    def run_extract_global_features(self, a):
        featuress = []
        lengths = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    features = f['features'][:]
                    lengths.append(len(features))
                    featuress.append(features)

        features = np.concatenate(featuress)

        with h5py.File('data/global_features.h5', 'w') as f:
            f.create_dataset('global_features', data=features)
            f.create_dataset('lengths', data=np.array(lengths))

    def run_extract_slide_features(self, a):
        data = []
        features = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    features.append(f['slide_feature'][:])
                data.append({
                    'name': name,
                    'order': i,
                    'filename': os.path.basename(h5_path),
                })

        df = pd.DataFrame(data)
        features = np.array(features)
        print('features', features.shape)

        o = 'data/slide_features.h5'
        with h5py.File(o, 'w') as f:
            f.create_dataset('features', data=features)
            f.create_dataset('names', data=df['name'].values)
            f.create_dataset('orders', data=df['order'].values)
            f.create_dataset('filenames', data=df['filename'].values)
        print(f'wrote {o}')

    class GlobalClusterArgs(CommonArgs):
        noshow: bool = False
        n_samples: int = Field(100, s='-N')

    def run_global_cluster(self, a):
        features = []
        images = []
        dfs = []
        for dir in sorted(glob('data/dataset/*')):
            name = os.path.basename(dir)
            for i, h5_path in enumerate(sorted(glob(f'{dir}/*.h5'))):
                with h5py.File(h5_path, 'r') as f:
                    patch_count = f['metadata/patch_count'][()]
                    ii = np.random.choice(patch_count, size=a.n_samples, replace=False)
                    ii = np.sort(ii)
                    features.append(f['features'][ii])
                    images.append(f['patches'][ii])
                    df_wsi = pd.DataFrame({'index': ii})
                df_wsi['name'] = int(os.path.basename(dir))
                df_wsi['order'] = i
                df_wsi['filename'] = os.path.basename(h5_path)
                dfs.append(df_wsi)

        df = pd.concat(dfs)
        df_clinical = pd.read_excel('./data/clinical_data_cleaned.xlsx', index_col=0)
        df = pd.merge(
            df,
            df_clinical,
            left_on='name',
            right_index=True,
            how='left'
        )

        features = np.concatenate(features)
        images = np.concatenate(images)
        # images = [Image.fromarray(i) for i in images]

        print('Loaded features', features.dtype, features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('UMAP fitting...')
        reducer = umap.UMAP(
                # n_neighbors=80,
                # min_dist=0.3,
                n_components=2,
                metric='cosine',
                min_dist=0.5,
                spread=2.0
                # random_state=a.seed
            )
        embedding = reducer.fit_transform(scaled_features)
        print('UMAP ok')

        # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=df['LDH'].values)
        # hover_images_on_scatters([scatter], [images])

        target = 'HANS'

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0)
        for (x, y), image, (_idx, row) in zip(embedding, images, df.iterrows()):
            img = OffsetImage(image, zoom=.125)
            value = row[target]

            text = TextArea(row['name'], textprops=dict(color='#000', ha='center'))
            vpack = VPacker(children=[text, img], align='center', pad=1)

            cmap = plt.cm.viridis
            color = '#333' if value < 0 else cmap(value)
            bbox_props = dict(boxstyle='square,pad=0.1', edgecolor=color, linewidth=1, facecolor='none')

            ab = AnnotationBbox(vpack, (x, y), frameon=True, bboxprops=bbox_props)
            ax.add_artist(ab)

        plt.title(f'UMAP')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    cli = CLI()
    cli.run()
