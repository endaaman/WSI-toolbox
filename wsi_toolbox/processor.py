import os
import gc
import sys

from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
import h5py
from openslide import OpenSlide
import tifffile
import zarr
import torch
import timm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import networkx as nx
import leidenalg as la
import igraph as ig

from .utils import find_optimal_components, create_frame, get_platform_font
from .utils.progress import tqdm_or_st


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

def cosine_distance(x, y):
    distance = np.linalg.norm(x - y)
    weight = np.exp(-distance / distance.mean())
    return distance, weight

class WSIFile:
    def __init__(self, path):
        pass

    def get_mpp(self):
        pass

    def get_original_size(self):
        pass

    def read_region(self, xywh):
        pass


class WSITiffFile(WSIFile):
    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)

        store = self.tif.pages[0].aszarr()
        self.zarr_data = zarr.open(store, mode='r')  # 読み込み専用で開く

    def get_original_size(self):
        s = self.tif.pages[0].shape
        return (s[1], s[0])

    def get_mpp(self):
        tags = self.tif.pages[0].tags
        resolution_unit = tags.get('ResolutionUnit', None)
        x_resolution = tags.get('XResolution', None)

        assert resolution_unit
        assert x_resolution

        x_res_value = x_resolution.value
        if isinstance(x_res_value, tuple) and len(x_res_value) == 2:
            # 分数の形式（分子/分母）
            numerator, denominator = x_res_value
            resolution = numerator / denominator
        else:
            resolution = x_res_value

        # 解像度単位の判定（2=インチ、3=センチメートル）
        if resolution_unit.value == 2:  # インチ
            # インチあたりのピクセル数からミクロンあたりのピクセル数へ変換
            # 1インチ = 25400ミクロン
            mpp = 25400.0 / resolution
        elif resolution_unit.value == 3:  # センチメートル
            # センチメートルあたりのピクセル数からミクロンあたりのピクセル数へ変換
            # 1センチメートル = 10000ミクロン
            mpp = 10000.0 / resolution
        else:
            mpp = 1.0 / resolution  # 単位不明の場合

        return mpp

    def read_region(self, xywh):
        x, y, width, height = xywh
        page = self.tif.pages[0]

        full_width = page.shape[1]  # tifffileでは[height, width]の順
        full_height = page.shape[0]

        x = max(0, min(x, full_width - 1))
        y = max(0, min(y, full_height - 1))
        width = min(width, full_width - x)
        height = min(height, full_height - y)

        if page.is_tiled:
            # LLMに聞くと region 引数が現れるがそんなものはない
            # region = page.asarray(region=(y, x, height, width))
            region = self.zarr_data[y:y+height, x:x+width]
        else:
            full_image = page.asarray()
            region = full_image[y:y+height, x:x+width]

        # カラーモデルの処理
        if region.ndim == 2:  # グレースケール
            region = np.stack([region, region, region], axis=-1)
        elif region.shape[2] == 4:  # RGBA
            region = region[:, :, :3]  # RGBのみ取得
        return region


class WSIOpenSlideFile(WSIFile):
    def __init__(self, path):
        self.wsi = OpenSlide(path)
        self.prop = dict(self.wsi.properties)

    def get_mpp(self):
        return float(self.prop['openslide.mpp-x'])

    def get_original_size(self):
        dim = self.wsi.level_dimensions[0]
        return (dim[0], dim[1])

    def read_region(self, xywh):
        # self.wsi.read_region((0, row*T), target_level, (width, T))
        # self.wsi.read_region((x, y), target_level, (w, h))
        img = self.wsi.read_region((xywh[0], xywh[1]), 0, (xywh[2], xywh[3])).convert('RGB')
        img = np.array(img.convert('RGB'))
        return img


class WSIProcessor:
    wsi: WSIFile
    def __init__(self, wsi_path, engine='auto'):
        if engine == 'auto':
            ext = os.path.splitext(wsi_path)[1]
            if ext == '.ndpi':
                engine = 'tifffile'
            else:
                engine = 'openslide'
        self.engine = engine
        if engine == 'openslide':
            self.wsi = WSIOpenSlideFile(wsi_path)
        elif engine == 'tifffile':
            self.wsi = WSITiffFile(wsi_path)
        else:
            raise ValueError('Invalid engine', a.engine)
        self.target_level = 0
        self.original_mpp = self.wsi.get_mpp()

        if 0.360 < self.original_mpp < 0.500:
            self.scale = 1
        elif self.original_mpp < 0.360:
            self.scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        self.mpp = self.original_mpp * self.scale


    def convert_to_hdf5(self, hdf5_path, patch_size=256, progress='tqdm'):
        S = patch_size   # Scaled patch size
        T = S*self.scale # Original patch size
        W, H = self.wsi.get_original_size()
        x_patch_count = W//T
        y_patch_count = H//T
        width = (W//T)*T
        row_count = H//T
        coordinates = []
        total_patches = []

        if progress == 'tqdm':
            print('Target level', self.target_level)
            print(f'Original mpp: {self.original_mpp:.6f}')
            print(f'Image mpp: {self.mpp:.6f}')
            print('Targt resolutions', W, H)
            print('Obtained resolutions', x_patch_count*S, y_patch_count*S)
            print('Scale', self.scale)
            print('Patch size', T)
            print('Scaled patch size', S)
            print('row count:', y_patch_count)
            print('col count:', x_patch_count)

        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('metadata/original_mpp', data=self.original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=self.target_level)
            f.create_dataset('metadata/mpp', data=self.mpp)
            f.create_dataset('metadata/scale', data=self.scale)
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
            tq = tqdm_or_st(range(row_count), backend=progress)
            for row in tq:
                image = self.wsi.read_region((0, row*T, width, T))
                image = cv2.resize(image, (width//self.scale, S), interpolation=cv2.INTER_LANCZOS4)

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
                tq.set_description(f'Selected patch count {len(batch)}/{len(patches)} ({row}/{y_patch_count})')
                tq.refresh()

            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        if progress == 'tqdm':
            print(f'{len(coordinates)} patches were selected.')


class TileProcessor:
    def __init__(self, model_name='gigapath', device='cuda'):
        assert model_name in ['uni', 'gigapath']
        self.model_name = model_name
        self.device = device
        self.target_name = f'{model_name}/features'

    def evaluate_hdf5_file(self, hdf5_path, batch_size=256, overwrite=False, progress='tqdm'):
        if self.model_name == 'uni':
            model = timm.create_model('hf-hub:MahmoodLab/uni',
                                      pretrained=True,
                                      dynamic_img_size=True,
                                      init_values=1e-5)
        elif self.model_name == 'gigapath':
            model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                      pretrained=True,
                                      dynamic_img_size=True)
        else:
            raise ValueError('Invalid model_name', self.model_name)  # model_nameをself.model_nameに修正

        model = model.eval().to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        done = False

        with h5py.File(hdf5_path, 'r+') as f:  # 'r+'に変更して読み書き両方可能に
            try:
                if self.target_name in f:
                    if overwrite:
                        print('Overwriting features.')
                        del f[self.target_name]
                    else:
                        return

                patch_count = f['metadata/patch_count'][()]
                batch_idx = [
                    (i, min(i+batch_size, patch_count))
                    for i in range(0, patch_count, batch_size)
                ]

                # バッファに全部メモリを確保せず、直接書き込む
                # データセットを前もって作成
                f.create_dataset(self.target_name, shape=(patch_count, model.num_features), dtype=np.float32)

                tq = tqdm_or_st(batch_idx, backend=progress)
                for i0, i1 in tq:
                    coords = f['coordinates'][i0:i1]
                    x = f['patches'][i0:i1]
                    x = (torch.from_numpy(x)/255).permute(0, 3, 1, 2) # BHWC->BCHW
                    x = x.to(self.device)
                    x = (x-mean)/std

                    with torch.no_grad():
                        h = model(x)

                    h_np = h.cpu().detach().numpy()

                    f[self.target_name][i0:i1] = h_np

                    # 明示的にGPUメモリを解放
                    del x, h
                    torch.cuda.empty_cache()
                    tq.set_description(f'Processing {i0}-{i1}(total={patch_count})')
                    tq.refresh()

                print('embeddings dimension', f[self.target_name].shape)
                done = True

            finally:
                if not done:
                    del f[self.target_name]
                    print(f'ABORTED! deleted {self.target_name}')
                del model, mean, std
                torch.cuda.empty_cache()
                gc.collect()




class ClusterProcessor:
    def __init__(self, hdf5_paths, model_name='gigapath', cluster_name=''):
        assert model_name in ['uni', 'gigapath']
        self.multi = len(hdf5_paths) > 1
        if self.multi:
            if not cluster_name:
                raise RuntimeError('Multiple files provided but name was not specified.')

        self.hdf5_paths = hdf5_paths
        self.model_name = model_name
        self.cluster_name = cluster_name

        if self.multi:
            self.clusters_path = f'{self.model_name}/clusters_{self.cluster_name}'
        else:
            self.clusters_path = f'{self.model_name}/clusters'

        features = []
        lengths = []
        clusters = []
        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                patch_count = f['metadata/patch_count'][()]
                lengths.append(patch_count)
                features.append(f[f'{self.model_name}/features'][:])
                if self.clusters_path in f:
                    clusters.append(f[self.clusters_path][:])

        self.lengths = lengths
        features = np.concatenate(features)
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(features)

        self.clusters = np.concatenate(clusters) if len(clusters)>0 else None

        self._umap_embeddings = None

    def get_umap_embeddings(self):
        if np.any(self._umap_embeddings):
            return self._umap_embeddings

        print('UMAP fitting...')
        reducer = umap.UMAP(
                # n_neighbors=30,
                # min_dist=0.05,
                n_components=2,
                # random_state=a.seed
            )
        embs = reducer.fit_transform(self.scaled_features)
        print('UMAP done')
        self._umap_embeddings = embs
        return embs


    def anlyze_clusters(self, resolution=1.0, use_umap_embs=False, overwrite=False, progress='tqdm'):
        if np.any(self.clusters) and not overwrite:
            print('Skip clustering')
            return

        n_samples = self.scaled_features.shape[0]
        tq = tqdm_or_st(total=n_samples+3, backend=progress)

        tq.set_description(f'Processing PCA...')
        n_components = find_optimal_components(self.scaled_features)
        print('Optimal n_components:', n_components)
        pca = PCA(n_components)
        target_features = pca.fit_transform(self.scaled_features)
        tq.update(1)

        tq.set_description(f'Processing KNN')
        k = int(np.sqrt(len(target_features)))
        nn = NearestNeighbors(n_neighbors=k).fit(target_features)
        distances, indices = nn.kneighbors(target_features)
        tq.update(1)

        G = nx.Graph()
        G.add_nodes_from(range(n_samples))

        h = self.get_umap_embeddings() if use_umap_embs else target_features
        print('umap_embeddings', use_umap_embs)
        tq.set_description(f'Processing edges...')
        for i in range(n_samples):
            for j in indices[i]:
                if i == j: # skip self loop
                    continue
                if use_umap_embs:
                    distance = np.linalg.norm(h[i] - h[j])
                    weight = np.exp(-distance)
                else:
                    explained_variance_ratio = pca.explained_variance_ratio_
                    weighted_diff = (h[i] - h[j]) * np.sqrt(explained_variance_ratio[:len(h[i])])
                    distance = np.linalg.norm(weighted_diff)
                    weight = np.exp(-distance / distance.mean())
                G.add_edge(i, j, weight=weight)
            tq.update(1)

        tq.set_description(f'Leiden clustering...')
        edges = list(G.edges())
        weights = [G[u][v]['weight'] for u, v in edges]
        ig_graph = ig.Graph(n=n_samples, edges=edges, edge_attrs={'weight': weights})

        print('Starting leiden clustering...')
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution, # maybe most adaptive
            # resolution_parameter=1.0, # maybe most adaptive
            # resolution_parameter=0.5, # more coarse cluster
        )
        tq.update(1)
        tq.close()
        print('leiden clustering done')

        # Convert partition result to cluster assignments
        clusters = np.full(n_samples, -1)  # Initialize all as noise
        for i, community in enumerate(partition):
            for node in community:
                clusters[node] = i

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        # print('n_clusters', n_clusters)
        # print('n_noise', n_noise)

        cursor = 0
        for hdf5_path, length in zip(self.hdf5_paths, self.lengths):
            cc = clusters[cursor:cursor+length]
            cursor += length
            with h5py.File(hdf5_path, 'a') as f:
                if self.clusters_path in f:
                    del f[self.clusters_path]
                f.create_dataset(self.clusters_path, data=cc)

        self.clusters = clusters


    def save_umap(self, fig_path):
        if not np.any(self.clusters):
            raise RuntimeError('Compute clusters before umap projection.')
        clusters = self.clusters
        cluster_ids = sorted(list(set(clusters)))

        umap_embeddings = self.get_umap_embeddings()
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.get_cmap('tab20')

        for i, cluster_id in enumerate(cluster_ids):
            coords = umap_embeddings[clusters == cluster_id]
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
            cluster_points = umap_embeddings[clusters == cluster_id]
            if len(cluster_points) < 1:
                continue
            centroid_x = np.mean(cluster_points[:, 0])
            centroid_y = np.mean(cluster_points[:, 1])
            ax.text(centroid_x, centroid_y, str(cluster_id),
                   fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))

        plt.title(f'UMAP + Clustering')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.savefig(fig_path)
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.5)
        print(f'wrote {fig_path}')
        return fig_path


class ThumbProcessor:
    def __init__(self, hdf5_path, cluster_name='', size=64):
        self.hdf5_path = hdf5_path
        self.cluster_name = cluster_name
        self.size = size

    def create_thumbnail(self, output_path, progress='tqdm'):
        S = self.size

        cmap = plt.get_cmap('tab20')
        with h5py.File(self.hdf5_path, 'r') as f:
            cols = f['metadata/cols'][()]
            rows = f['metadata/rows'][()]
            patch_count = f['metadata/patch_count'][()]
            patch_size = f['metadata/patch_size'][()]

            clusters = []
            cluster_path = 'gigapath/clusters'
            if self.cluster_name:
                cluster_path += f'_{self.cluster_name}'
            clusters = f[cluster_path][:]

            frames = {}
            font = ImageFont.truetype(font=get_platform_font(), size=16)
            for cluster in np.unique(clusters).tolist() + [-1]:
                if cluster < 0:
                    color = '#111'
                else:
                    color = mcolors.rgb2hex(cmap(cluster)[:3])
                frames[cluster] = create_frame(S, color, f'{cluster}', font)

            canvas = Image.new('RGB', (cols*S, rows*S), (0,0,0))
            tq = tqdm_or_st(range(patch_count), backend=progress)
            for i in tq:
                coord = f['coordinates'][i]
                x, y = coord//patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch)
                patch = patch.resize((S, S))
                frame = frames[clusters[i]]
                patch.paste(frame, (0, 0), frame)
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(output_path)
            print(f'wrote {output_path}')
