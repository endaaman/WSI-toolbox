import os
import sys
import warnings
from glob import glob
from tqdm import tqdm
from pydantic import Field
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

from .processor import WSIProcessor, TileProcessor, ClusterProcessor, PreviewClustersProcessor
from .common import create_model
from .utils import hover_images_on_scatters, find_optimal_components, create_frame, get_platform_font
from .utils.cli import BaseMLCLI, BaseMLArgs
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
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        patch_size: int = 256
        overwrite: bool = Field(False, s='-O')
        engine: str = Field('auto', choices=['auto', 'openslide', 'tifffile'])
        mpp: float = 0

    def run_wsi2h5(self, a):
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


    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        cluster_name: str = Field('', l='--name', s='-N')
        size: int = 64
        model: str = Field('gigapath', choice=['gigapath', 'uni', 'unified', 'none'])
        open: bool = False

    def run_preview(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            if a.cluster_name:
                output_path = f'{base}_thumb_{a.cluster_name}.jpg'
            else:
                output_path = f'{base}_thumb.jpg'

        thumb_proc = PreviewClustersProcessor(
                a.input_path,
                cluster_name=a.cluster_name,
                size=a.size)
        img = thumb_proc.create_thumbnail(progress='tqdm')
        img.save(output_path)

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        name: str
        size: int = 64
        model: str = Field('gigapath', choice=['gigapath', 'uni', 'unified', 'none'])
        open: bool = False

    def run_preview_scores(self, a):
        S = a.size
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_scores_{a.name}.jpg'

        cmap = plt.get_cmap('viridis')
        font = ImageFont.truetype(font=get_platform_font(), size=12)

        with h5py.File(a.input_path, 'r') as f:
            cols = f['metadata/cols'][()]
            rows = f['metadata/rows'][()]
            patch_count = f['metadata/patch_count'][()]
            patch_size = f['metadata/patch_size'][()]
            coordinates = f['coordinates'][()]
            scores = f[f'{a.model}/scores_{a.name}'][()]

            print('Filtered scores:', scores[scores > 0].shape)

            canvas = Image.new('RGB', (cols*S, rows*S), (0,0,0))
            for i in tqdm(range(patch_count)):
                coord = coordinates[i]
                x, y = coord//patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch)
                patch = patch.resize((S, S))
                score = scores[i]
                if not np.isnan(score):
                    color = mcolors.rgb2hex(cmap(score)[:3])
                    frame = create_frame(S, color, f'{score:.3f}', font)
                    patch.paste(frame, (0, 0), frame)
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(output_path)
            print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class ProcessPatchesArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        batch_size: int = Field(512, s='-B')
        overwrite: bool = Field(False, s='-O')
        model_name: str = Field('gigapath', choice=['gigapath', 'uni'], l='--model', s='-M')
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

    def run_process_slide(self, a):
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
        name: str = ''
        model: str = Field('gigapath', choices=['gigapath', 'uni'])
        resolution: float = 1
        use_umap_embs: float = False
        nosave: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s='-O')

    def run_cluster(self, a):
        cluster_proc = ClusterProcessor(
                a.input_paths,
                model_name=a.model,
                cluster_name=a.name)
        cluster_proc.anlyze_clusters(
                resolution=a.resolution,
                use_umap_embs=a.use_umap_embs,
                overwrite=a.overwrite,
                progress='tqdm')

        if len(a.input_paths) > 1:
            # multiple
            dir = os.path.dirname(a.input_paths[0])
            fig_path = f'{dir}/{a.name}.png'
        else:
            base, ext = os.path.splitext(a.input_paths[0])
            fig_path = f'{base}_umap.png'

        fig = cluster_proc.plot_umap()
        if not a.nosave:
            fig.savefig(fig_path)
            print(f'wrote {fig_path}')

        if not a.noshow:
            plt.show()


    class ClusterScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        name: str = Field(...)
        target_clusters: list[int] = Field(..., s='-T')
        model: str = Field('gigapath', l='--cluster', s='-C', choice=['gigapath', 'uni', 'unified', 'none'])
        scaler: str = Field('minmax', choices=['std', 'minmax'])
        noshow: bool = False
        fig: str = ''

    def run_cluster_scores(self, a):
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            clusters = f[f'{a.model}/clusters'][:]
            mask = np.isin(clusters, a.target_clusters)
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

        for target in a.target_clusters:
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
            if a.fig:
                plt.savefig(a.fig)
            plt.show()


    class AlignKeysArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')

    def run_align_keys(self, a):
        with h5py.File(a.input_path, 'a') as f:
            if 'features' in f:
                features = f['features'][:]
                if 'gigapath/features' in f:
                    print('"gigapath/features" already exists')
                else:
                    f.create_dataset('gigapath/features', data=features)
                    print('"gigapath/features" was added')
                del f['features']
                print('"features" has been deleted')
            else:
                if 'gigapath/features' in f:
                    print('features OK')
                else:
                    print('Both "features" and "gigapath/features" do not exist')

            if 'slide_feature' in f:
                slide_feature = f['slide_feature'][:]
                if 'gigapath/slide_feature' in f:
                    print('"gigapath/slide_feature" already exists')
                else:
                    f.create_dataset('gigapath/slide_feature', data=slide_feature)
                    print('"gigapath/slide_feature" was added')

                del f['slide_feature']
                print('"slide_feature" has been deleted')
            else:
                if 'gigapath/slide_feature' in f:
                    print('slide_feature OK')
                else:
                    print('Both "slide_feature" and "gigapath/slide_feature" do not exist')

        # img  = (img*256).astype(np.uint8)
        # Image.fromarray(img).resize((256, 256), Image.NEAREST).save('1.png')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
