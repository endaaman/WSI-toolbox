import os
import sys
import warnings
from glob import glob
from pathlib import Path as P

from tqdm import tqdm
from pydantic import Field
from pydantic_autocli import param
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
import seaborn as sns
import h5py
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
import leidenalg as la
import igraph as ig
import hdbscan
import torch
from torchvision import transforms
from torch.amp import autocast
import timm
from gigapath import slide_encoder

from .processor import WSIProcessor, TileProcessor, ClusterProcessor, \
        PreviewClustersProcessor, PreviewScoresProcessor, PreviewLatentPCAProcessor, PreviewLatentClusterProcessor
from .common import DEFAULT_MODEL, create_model
from .utils import plot_umap
from .utils.cli import BaseMLCLI, BaseMLArgs
from .utils.analysis import leiden_cluster
from .utils.progress import tqdm_or_st


warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass

    class Wsi2h5Args(CommonArgs):
        input_path: str = param(..., l='--in', s='-i')
        output_path: str = param('', l='--out', s='-o')
        patch_size: int = param(256, s='-S')
        overwrite: bool = param(False, s='-O')
        engine: str = param('auto', choices=['auto', 'openslide', 'tifffile'])
        mpp: float = 0

    def run_wsi2h5(self, a:Wsi2h5Args):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = base + '.h5'

        if os.path.exists(output_path):
            if not a.overwrite:
                print(f'{output_path} exists. Skipping.')
                return
            print(f'{output_path} exists but overwriting it.')

        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        p = WSIProcessor(a.input_path, engine=a.engine, mpp=a.mpp)
        p.convert_to_hdf5(output_path, patch_size=a.patch_size, progress='tqdm')

        print('done')


    class ProcessPatchesArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        batch_size: int = Field(512, s='-B')
        overwrite: bool = Field(False, s='-O')
        model_name: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni'], l='--model', s='-M')
        with_latent_features: bool = Field(False, s='-L')

    def run_process_patches(self, a):
        tp = TileProcessor(model_name=a.model_name, device='cuda')
        tp.evaluate_hdf5_file(a.input_path,
                              batch_size=a.batch_size,
                              with_latent_features=a.with_latent_features,
                              overwrite=a.overwrite,
                              progress='tqdm')


    class ProcessSlideArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        overwrite: bool = Field(False, s='-O')

    def run_process_slide(self, a:ProcessSlideArgs):
        with h5py.File(a.input_path, 'r') as f:
            if 'slide_feature' in f:
                if not a.overwrite:
                    print('feature embeddings are already obtained.')
                    return
            features = f['gigapath/features'][:]
            coords = f['coordinates'][:]

        features = torch.tensor(features, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, D)
        coords = torch.tensor(coords, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, 2)

        print('Loading LongNet...')
        long_net = slide_encoder.create_model(
            'data/slide_encoder.pth',
            'gigapath_slide_enc12l768d',
            1536,
        ).eval().to(a.device)

        print('LongNet loaded.')

        with torch.set_grad_enabled(False):
            with autocast(a.device, dtype=torch.float16):
                output = long_net(features, coords)
            # output = output.cpu().detach()
            slide_feature = output[0][0].cpu().detach()

        print('slide_feature dimension:', slide_feature.shape)

        with h5py.File(a.input_path, 'a') as f:
            if a.overwrite and 'slide_feature' in f:
                print('Overwriting slide_feature.')
                del f['slide_feature']
            f.create_dataset('slide_feature', data=slide_feature)


    class ClusterArgs(CommonArgs):
        input_paths: list[str] = Field(..., l='--in', s='-i')
        cluster_name: str = Field('', l='--name', s='-n')
        sub: list[int] = Field([], l='--sub', s='-s')
        model: str = Field(DEFAULT_MODEL, choices=['gigapath', 'uni', 'virchow2'])
        resolution: float = 1
        use_umap_embs: float = False
        nosave: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s='-O')

    def run_cluster(self, a:ClusterArgs):
        cluster_proc = ClusterProcessor(
                a.input_paths,
                model_name=a.model,
                cluster_name=a.cluster_name,
                cluster_filter=a.sub,
                )
        cluster_proc.anlyze_clusters(
                resolution=a.resolution,
                use_umap_embs=a.use_umap_embs,
                overwrite=a.overwrite,
                progress='tqdm')

        if len(a.input_paths) > 1:
            # multiple
            dir = os.path.dirname(a.input_paths[0])
            base = fig_path = f'{dir}/{a.name}'
        else:
            base, ext = os.path.splitext(a.input_paths[0])

        s = ''
        if len(a.sub) > 0:
            s = 'sub-' + '-'.join([str(i) for i in a.sub]) + '_'
        fig_path = f'{base}_{s}umap.png'

        fig = cluster_proc.plot_umap()
        if not a.nosave:
            fig.savefig(fig_path)
            print(f'wrote {fig_path}')

        if not a.noshow:
            plt.show()


    class ClusterScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        name: str = Field(...)
        clusters: list[int] = Field([], s='-C')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        scaler: str = Field('minmax', choices=['std', 'minmax'])
        noshow: bool = False
        nosave: bool = False

    def run_cluster_scores(self, a:ClusterScoresArgs):
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            clusters = f[f'{a.model}/clusters'][:]
            mask = np.isin(clusters, a.clusters)
            masked_clusters = clusters[mask]
            masked_features = f[f'{a.model}/features'][mask]

        pca = PCA(n_components=1)
        values = pca.fit_transform(masked_features)

        if a.scaler == 'minmax':
            scaler = MinMaxScaler()
            values = scaler.fit_transform(values)
        elif a.scaler == 'std':
            scaler = StandardScaler()
            values = scaler.fit_transform(values)
            values = sigmoid(values)
        else:
            raise ValueError('Invalid scaler:', a.scaler)

        data = []
        labels = []

        for target in a.clusters:
            cluster_values = values[masked_clusters == target].flatten()
            data.append(cluster_values)
            labels.append(f'Cluster {target}')

        with h5py.File(a.input_path, 'a') as f:
            path = f'{a.model}/scores_{a.name}'
            if path in f:
                del f[path]
                print(f'Deleted {path}')
            vv = np.full(patch_count, np.nan, dtype=values.dtype)
            vv[mask] = values[:, 0]
            f[path] = vv
            print(f'Wrote {path} in {a.input_path}')

        if not a.noshow:
            plt.figure(figsize=(12, 8))
            sns.set_style('whitegrid')
            ax = plt.subplot(111)
            sns.violinplot(data=data, ax=ax, inner='box', cut=0, zorder=1, alpha=0.5)  # cut=0で分布全体を表示

            for i, d in enumerate(data):
                x = np.random.normal(i, 0.05, size=len(d))
                ax.scatter(x, d, alpha=.8, s=5, color=f'C{i}', zorder=2)

            ax.set_xticks(np.arange(0, len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('PCA Values')
            ax.set_title('Distribution of PCA Values by Cluster')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            if not a.nosave:
                p = P(a.input_path)
                fig_path = str(p.parent / f'{p.stem}_score-{a.name}_pca.png')
                plt.savefig(fig_path)
                print(f'wrote {fig_path}')
            plt.show()


    class ClusterLatentArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        name: str = ''
        model: str = Field(DEFAULT_MODEL, choices=['gigapath', 'uni', 'virchow2'])
        resolution: float = 1
        use_umap_embs: float = False
        nosave: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s='-O')

    def run_cluster_latent(self, a):
        target_path = f'{a.model}/latent_clusters'
        skip = False
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            features = f[f'{a.model}/latent_features'][:]
            if target_path in f:
                if a.overwrite:
                    print(f'overwriting old {target_path} of {a.input_path}')
                else:
                    skip = True
                    clusters = f[target_path][:]
                    # raise RuntimeError(f'{target_path} already exists in {a.input_path}')

        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)
        s = features.shape
        h = features.reshape(s[0]*s[1], s[-1]) # B*16*16, 3

        if not skip:
            clusters = leiden_cluster(h,
                                      umap_emb_func=None,
                                      resolution=a.resolution,
                                      n_jobs=-1,
                                      progress='tqdm')

            clusters = clusters.reshape(s[0], s[1])

            with h5py.File(a.input_path, 'a') as f:
                if target_path in f:
                    del f[target_path]
                f.create_dataset(target_path, data=clusters)

        print(features.reshape(s[0]*s[1], -1).shape)
        print(clusters.reshape(s[0]*s[1]).shape)

        reducer = umap.UMAP(
                # n_neighbors=30,
                # min_dist=0.05,
                n_components=2,
                # random_state=a.seed
            )

        embs = reducer.fit_transform(features.reshape(s[0]*s[1], -1))
        fig = plot_umap(
                embeddings=embs,
                clusters=clusters.reshape(s[0]*s[1]))
        if not a.nosave:
            p = P(a.input_path)
            fig_path = str(p.parent / f'{p.stem}_latent_umap.png')
            plt.savefig(fig_path)
            print(f'wrote {fig_path}')
        plt.show()


    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'virchow2'])
        cluster_name: str = Field('', l='--name', s='-N')
        size: int = 64
        open: bool = False

    def run_preview(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            if a.cluster_name:
                output_path = f'{base}_{a.cluster_name}_thumb.jpg'
            else:
                output_path = f'{base}_thumb.jpg'

        proc = PreviewClustersProcessor(
                a.input_path,
                model_name=a.model,
                size=a.size)
        img = proc.create_thumbnail(
                cluster_name=a.cluster_name,
                progress='tqdm')
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'unified', 'none'])
        score_name: str = Field(..., l='--name', s='-N')
        size: int = 64
        open: bool = False

    def run_preview_scores(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_score-{a.score_name}_thumb.jpg'

        proc = PreviewScoresProcessor(
                a.input_path,
                model_name=a.model,
                size=a.size)
        img = proc.create_thumbnail(
                score_name=a.score_name,
                progress='tqdm')
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewLatentPcaArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        alpha: float = 0.5
        size: int = 64
        open: bool = False

    def run_preview_latent_pca(self, a:PreviewArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_latent_pca.jpg'

        proc = PreviewLatentPCAProcessor(
                a.input_path,
                model_name=a.model,
                size=a.size)
        img = proc.create_thumbnail(
                progress='tqdm')
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewLatentArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        alpha: float = 0.5
        size: int = 64
        open: bool = False

    def run_preview_latent(self, a:PreviewArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_latent_clusters.jpg'

        proc = PreviewLatentClusterProcessor(
                a.input_path,
                model_name=a.model,
                size=a.size)
        img = proc.create_thumbnail(
                progress='tqdm')
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')



if __name__ == '__main__':
    cli = CLI()
    cli.run()
