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

from .wsi import WSIOpenSlideFile, WSITiffFile
from .utils import hover_images_on_scatters
from .utils.cli import BaseMLCLI, BaseMLArgs
from .utils.progress import tqdm_or_st

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")





def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def is_white_patch(patch, rgb_std_threshold=7.0, white_ratio=0.7):
    # white: RGB std < 7.0
    rgb_std_pixels = np.std(patch, axis=2) < rgb_std_threshold
    white_pixels = np.sum(rgb_std_pixels)
    total_pixels = patch.shape[0] * patch.shape[1]
    white_ratio_calculated = white_pixels / total_pixels
    # print('whi' if white_ratio_calculated > white_ratio else 'use',
    #       'std{:.3f}'.format(np.sum(rgb_std_pixels)/total_pixels)
    #      )
    return white_ratio_calculated > white_ratio


def find_optimal_components(features, threshold=0.95):
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_ratio_
    # 累積寄与率が95%を超える次元数を選択する例
    cumulative_variance = np.cumsum(explained_variance)
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1
    return min(optimal_n, len(features) - 1)


def get_platform_font():
    if sys.platform == 'win32':
        # Windows
        font_path = 'C:\\Windows\\Fonts\\msgothic.ttc'  # MSゴシック
    elif sys.platform == 'darwin':
        # macOS
        font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    else:
        # Linux
        # font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' # TODO: propagation
        font_path = '/usr/share/fonts/TTF/DejaVuSans.ttf'
    return font_path


def create_frame(size, color, text, font):
    frame = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, size, size), outline=color, width=4)
    text_color = 'white' if mcolors.rgb_to_hsv(mcolors.hex2color(color))[2]<0.9 else 'black'
    bbox = np.array(draw.textbbox((0, 0), text, font=font))
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle((4, 4, bbox[2]+4, bbox[3]+4), fill=color)
    draw.text((1, 1), text, font=font, fill=text_color)
    return frame



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass

    class Wsi2h5Args(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field(..., l='--out', s='-o')
        patch_size: int = 256
        overwrite: bool = False
        engine: str = Field('openslide', choices=['openslide', 'tifffile'])

    def run_wsi2h5(self, a):
        if os.path.exists(a.output_path):
            if not a.overwrite:
                print(f'{a.output_path} exists. Skipping.')
                return
            print(f'{a.output_path} exists but overwriting it.')

        wsi: WSIFile
        if a.engine == 'openslide':
            wsi = WSIOpenSlideFile(a.input_path)
        elif a.engine == 'tifffile':
            wsi = WSITiffFile(a.input_path)
        else:
            raise ValueError('Invalid engine', a.engine)

        original_mpp = wsi.get_mpp()
        target_level = 0

        if 0.360 < original_mpp < 0.500:
            scale = 1
        elif original_mpp < 0.360:
            scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        mpp = original_mpp * scale

        S = a.patch_size # scaled patch size
        T = S*scale      # actual patch size

        # dimension = wsi.level_dimensions[target_level]
        # W, H = dimension[0], dimension[1]
        W, H = wsi.get_original_size()

        x_patch_count = W//T
        y_patch_count = H//T
        width = (W//T)*T
        row_count = H//T

        print('Target level', target_level)
        print(f'Original mpp: {original_mpp:.6f}')
        print(f'Image mpp: {mpp:.6f}')
        print('Targt resolutions', W, H)
        print('Obtained resolutions', x_patch_count*S, y_patch_count*S)
        print('Scale', scale)
        print('Patch size', T)
        print('Scaled patch size', S)
        print('row count:', y_patch_count)
        print('col count:', x_patch_count)

        coordinates = []
        total_patches = []

        d = os.path.dirname(a.output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        with h5py.File(a.output_path, 'w') as f:
            f.create_dataset('metadata/original_mpp', data=original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=target_level)
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/scale', data=scale)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)

            total_patches = f.create_dataset(
                    'patches',
                    shape=(x_patch_count*y_patch_count, S, S, 3),
                    dtype=np.uint8,
                    chunks=(1, S, S, 3),
                    compression='gzip',
                    compression_opts=9)

            cursor = 0
            tq = tqdm(range(row_count))
            for row in tq:
                image = wsi.read_region((0, row*T, width, T))
                # image = image.resize((width//scale, S))
                image = cv2.resize(image, (width//scale, S), interpolation=cv2.INTER_LANCZOS4)

                patches = image.reshape(1, S, x_patch_count, S, 3) # (y, h, x, w, 3)
                patches = patches.transpose(0, 2, 1, 3, 4)   # (y, x, h, w, 3)
                patches = patches[0]

                batch = []
                for col, patch in enumerate(patches):
                    if is_white_patch(patch):
                        continue
                    # Image.fromarray(patch).save(f'out/{row}_{col}.jpg')
                    batch.append(patch)
                    coordinates.append((col*S, row*S))
                batch = np.array(batch)
                total_patches[cursor:cursor+len(batch), ...] = batch
                cursor += len(batch)
                tq.set_description(f'selected patch count {len(batch)}/{len(patches)} ({row}/{y_patch_count})')
                tq.refresh()

            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        print(f'{len(coordinates)} patches were selected.')
        print('done')

    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        size: int = 64
        model: str = Field('gigapath', choice=['gigapath', 'uni', 'unified', 'none'])
        open: bool = False

    def run_preview(self, a):
        S = a.size
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_thumb.jpg'

        cmap = plt.get_cmap('tab20')
        with h5py.File(a.input_path, 'r') as f:
            cols = f['metadata/cols'][()]
            rows = f['metadata/rows'][()]
            patch_count = f['metadata/patch_count'][()]
            patch_size = f['metadata/patch_size'][()]

            show_clusters = False
            clusters = []
            if a.model != 'none':
                if f'{a.model}/clusters' in f:
                    show_clusters = True
                    clusters = f[f'{a.model}/clusters'][:]
                    print('loaded cluster data', clusters.shape)
                else:
                    print(f'"{a.model}/clusters" was not found in h5 data.')

            frames = {}
            if show_clusters:
                font = ImageFont.truetype(font=get_platform_font(), size=16)
                for cluster in np.unique(clusters).tolist() + [-1]:
                    if cluster < 0:
                        color = '#111'
                    else:
                        color = mcolors.rgb2hex(cmap(cluster)[:3])
                    frames[cluster] = create_frame(S, color, f'{cluster}', font)

            canvas = Image.new('RGB', (cols*S, rows*S), (0,0,0))
            for i in tqdm(range(patch_count)):
                coord = f['coordinates'][i]
                x, y = coord//patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch)
                patch = patch.resize((S, S))
                if show_clusters:
                    frame = frames[clusters[i]]
                    patch.paste(frame, (0, 0), frame)
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(output_path)
            print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        size: int = 64
        model: str = Field('gigapath', choice=['gigapath', 'uni', 'unified', 'none'])
        open: bool = False

    def run_preview_scores(self, a):
        S = a.size
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_scores.jpg'

        cmap = plt.get_cmap('viridis')
        font = ImageFont.truetype(font=get_platform_font(), size=16)

        with h5py.File(a.input_path, 'r') as f:
            cols = f['metadata/cols'][()]
            rows = f['metadata/rows'][()]
            patch_count = f['metadata/patch_count'][()]
            patch_size = f['metadata/patch_size'][()]
            coordinates = f['coordinates'][()]
            scores = f[f'{a.model}/scores'][()]

            print('Filtered scores:', scores[scores > 0].shape)

            canvas = Image.new('RGB', (cols*S, rows*S), (0,0,0))
            for i in tqdm(range(patch_count)):
                coord = coordinates[i]
                x, y = coord//patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch)
                patch = patch.resize((S, S))
                score = scores[i]
                if score > 0:
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
        overwrite: bool = False
        model_name: str = Field('gigapath', choice=['gigapath', 'uni'], l='--model', s='-M')

    def run_process_patches(self, a):
        target_name = f'{a.model_name}/features'
        with h5py.File(a.input_path, 'r') as f:
            if target_name in f:
                if not a.overwrite:
                    print('patch embeddings are already obtained.')
                    return

        if a.model_name == 'uni':
            model = timm.create_model('hf-hub:MahmoodLab/uni',
                                      pretrained=True,
                                      dynamic_img_size=True,
                                      init_values=1e-5)
        elif a.model_name == 'gigapath':
            model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                      pretrained=True,
                                      dynamic_img_size=True)
        else:
            raise ValueError('Invalid model_name', a.model_name)

        model = model.eval().to(a.device)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(a.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(a.device)

        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            batch_idx = [
                (i, min(i+a.batch_size, patch_count))
                for i in range(0, patch_count, a.batch_size)
            ]
            hh = []
            for i0, i1 in tqdm(batch_idx):
                coords = f['coordinates'][i0:i1]
                x = f['patches'][i0:i1]
                x = (torch.from_numpy(x)/255).permute(0, 3, 1, 2) # BHWC->BCHW
                x = x.to(a.device)
                x = (x-mean)/std
                with torch.set_grad_enabled(False):
                    h = model(x)
                h = h.cpu().detach()
                hh.append(h)
            hh = torch.cat(hh).numpy()

        print('embeddings dimension', hh.shape)
        assert len(hh) == patch_count

        with h5py.File(a.input_path, 'a') as f:
            if a.overwrite and 'features' in f:
                print('Overwriting features.')
                del f[target_name]
            f.create_dataset(target_name, data=hh)


    class ProcessSlideArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        overwrite: bool = False

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
        input_path: str = Field(..., l='--in', s='-i')
        models: list[str] = Field(['gigapath'], choices=['uni', 'gigapath'])
        method: str = Field('leiden', s='-M')
        nosave: bool = False
        noshow: bool = False

    def run_cluster(self, a):
        assert len(a.models) > 0
        name = {
            frozenset(['uni']): 'uni',
            frozenset(['gigapath']): 'gigapath',
            frozenset(['uni', 'gigapath']): 'unified'
        }.get(frozenset(a.models))
        if not name:
            raise ValueError('Invalid models', a.models)

        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            feature_arrays = []
            print(f.keys())
            for model in a.models:
                path = f'{model}/features'
                if path in f:
                    feature_arrays.append(f[path][:])
                else:
                    raise RuntimeError(f'"{path}" does not exist. Do `process-patches` first')
            features = np.concatenate(feature_arrays, axis=1)

        print('Loaded features', features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('UMAP fitting...')
        reducer = umap.UMAP(
                # n_neighbors=30,
                # min_dist=0.05,
                n_components=2,
                # random_state=a.seed
            )
        embedding = reducer.fit_transform(scaled_features)
        print('UMAP done')

        eps = 0.2
        if a.method.lower() == 'leiden':
            # 13954, 18839
            n_components = find_optimal_components(scaled_features)
            print('Optimal n_components:', n_components)
            pca = PCA(n_components)
            target_features = pca.fit_transform(scaled_features)

            k = int(np.sqrt(len(target_features)))
            nn = NearestNeighbors(n_neighbors=k).fit(target_features)
            distances, indices = nn.kneighbors(target_features)

            G = nx.Graph()
            n_samples = embedding.shape[0]
            G.add_nodes_from(range(n_samples))

            # Add edges based on k-nearest neighbors
            for i in range(n_samples):
                for j in indices[i]:
                    if i != j:  # Avoid self-loops
                        # Add edge weight based on distance (closer points have higher weights)
                        distance = np.linalg.norm(embedding[i] - embedding[j])
                        weight = np.exp(-distance)  # Convert distance to similarity
                        G.add_edge(i, j, weight=weight)

            # Convert NetworkX graph to igraph for Leiden algorithm
            edges = list(G.edges())
            weights = [G[u][v]['weight'] for u, v in edges]
            ig_graph = ig.Graph(n=n_samples, edges=edges, edge_attrs={'weight': weights})

            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=1.0, # maybe most adaptive
                # resolution_parameter=0.5, # more coarse cluster
            )

            # Convert partition result to cluster assignments
            clusters = np.full(n_samples, -1)  # Initialize all as noise
            for i, community in enumerate(partition):
                for node in community:
                    clusters[node] = i

        elif a.method.lower() == 'hdbscan':
            target_features = embedding
            m = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=5,
                cluster_selection_epsilon=eps,
                metric='euclidean'
            )
            clusters = m.fit_predict(target_features)
        else:
            raise RuntimeError('Invalid medthod:', a.method)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print('n_clusters', n_clusters)
        print('n_noise', n_noise)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap('tab20')
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(set(clusters))))
        cluster_ids = sorted(list(set(clusters)))
        for i, cluster_id in enumerate(cluster_ids):
            coords = embedding[clusters == cluster_id]
            if cluster_id == -1:
                color = 'black'
                label = 'Noise'
                size = 12
            else:
                color = [cmap(cluster_id % 20)]
                label = f'Cluster {cluster_id}'
                size = 7
            plt.scatter(coords[:, 0], coords[:, 1], s=size, c=color, label=label)


        for cluster_id in cluster_ids:
            if cluster_id < 0:
                continue
            cluster_points = embedding[clusters == cluster_id]
            if len(cluster_points) < 1:
                continue
            centroid_x = np.mean(cluster_points[:, 0])
            centroid_y = np.mean(cluster_points[:, 1])
            ax.text(centroid_x, centroid_y, str(cluster_id),
                   fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))

        plt.title(f'UMAP + {a.method} Clustering')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if not a.nosave:
            with h5py.File(a.input_path, 'a') as f:
                path = f'{name}/clusters'
                if path in f:
                    del f[path]
                f.create_dataset(path, data=clusters)
            print(f'Save clusters to {a.input_path}')

            base, ext = os.path.splitext(a.input_path)
            fig_path = f'{base}_umap.png'
            plt.savefig(fig_path)
            print(f'wrote {fig_path}')

        if not a.noshow:
            plt.show()


    class ClusterScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        targets: list[int] = Field(..., s='-T')
        model: str = Field('gigapath', l='--cluster', s='-C', choice=['gigapath', 'uni', 'unified', 'none'])
        scaler: str = Field('minmax', choices=['std', 'minmax'])

    def run_cluster_scores(self, a):
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            clusters = f[f'{a.model}/clusters'][:]
            mask = np.isin(clusters, a.targets)
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

        for target in a.targets:
            cluster_values = values[masked_clusters == target].flatten()
            data.append(cluster_values)
            labels.append(f'Cluster {target}')

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
        plt.show()

        with h5py.File(a.input_path, 'a') as f:
            path = f'{a.model}/scores'
            if path in f:
                del f[path]
                print(f'Deleted {path}')
            vv = np.full(patch_count, -1, dtype=values.dtype)
            vv[mask] = values[:, 0]
            f[path] = vv
            print(f'Wrote {path} in {a.input_path}')


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



if __name__ == '__main__':
    cli = CLI()
    cli.run()
