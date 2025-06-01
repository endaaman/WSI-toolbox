import os
import gc
import sys
import warnings

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import networkx as nx
import leidenalg as la
import igraph as ig

from .common import create_model, DEFAULT_MODEL, DEFAULT_BACKEND
from .utils import create_frame, get_platform_font, plot_umap
from .utils.progress import tqdm_or_st
from .utils.analysis import leiden_cluster



warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


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

def safe_del(hdf_file, key_path):
    if key_path in hdf_file:
        del hdf_file[key_path]

class WSIFile:
    def __init__(self, path):
        pass

    def get_mpp(self):
        pass

    def get_original_size(self):
        pass

    def read_region(self, xywh):
        pass


class TiffFile(WSIFile):
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


class OpenSlideFile(WSIFile):
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


class StandardImage(WSIFile):
    def __init__(self, path, mpp):
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式で読み込むのでRGBに変換
        self.mpp = mpp
        assert self.mpp is not None, 'Specify mpp when using StandardImage'

    def get_mpp(self):
        return self.mpp

    def get_original_size(self):
        return self.image.shape[1], self.image.shape[0]  # width, height

    def read_region(self, xywh):
        x, y, w, h = xywh
        return self.image[y:y+h, x:x+w]

class WSIProcessor:
    wsi: WSIFile
    def __init__(self, image_path, engine='auto', **extra):
        if engine == 'auto':
            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.ndpi':
                engine = 'tifffile'
            elif ext in ['.jpg', '.jpeg', '.png', '.tif', 'tiff']:
                engine = 'standard'
            else:
                engine = 'openslide'
            print(f'using {engine} engine for {os.path.basename(image_path)}')

        self.engine = engine.lower()
        if engine == 'openslide':
            self.wsi = OpenSlideFile(image_path)
        elif engine == 'tifffile':
            self.wsi = TiffFile(image_path)
        elif engine == 'standard':
            mpp = extra.pop('mpp', None)
            self.wsi = StandardImage(image_path, mpp=mpp)
        else:
            raise ValueError('Invalid engine', engine)

        if extra:
            raise ValueError('Unprocessed extra arguments', extra)

        self.target_level = 0
        self.original_mpp = self.wsi.get_mpp()

        if 0.360 < self.original_mpp < 0.500:
            self.scale = 1
        elif self.original_mpp < 0.360:
            self.scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        self.mpp = self.original_mpp * self.scale


    def convert_to_hdf5(self, hdf5_path, patch_size=256, rotate=False, progress=DEFAULT_BACKEND):
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
                    if rotate:
                        patch = cv2.rotate(patch, cv2.ROTATE_180)
                        coordinates.append(((x_patch_count-1-col)*S, (y_patch_count-1-row)*S))
                    else:
                        coordinates.append((col*S, row*S))
                    batch.append(patch)
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
    def __init__(self, model_name=DEFAULT_MODEL, device='cuda'):
        assert model_name in ['uni', 'gigapath', 'virchow2']
        self.model_name = model_name
        self.device = device
        self.feature_name = f'{model_name}/features'
        self.latent_feature_name = f'{model_name}/latent_features'

    def evaluate_hdf5_file(self, hdf5_path, batch_size=256,
                           with_latent_features=False,
                           overwrite=False, progress=DEFAULT_BACKEND):
        model = create_model(self.model_name)
        model = model.eval().to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        done = False

        with h5py.File(hdf5_path, 'r+') as f:
            latent_size = model.patch_embed.proj.kernel_size[0]
            try:
                if overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.latent_feature_name)
                else:
                    if with_latent_features:
                        if (self.feature_name in f) and (self.latent_feature_name in f):
                            # Both exist
                            done = True # for finalization
                            print('Already extracted. Skipped.')
                            return
                        if (self.feature_name in f) or (self.latent_feature_name in f):
                            # Either exists
                            raise RuntimeError(f'Either {self.feature_name} or {self.latent_feature_name} exists.')
                    else:
                        if self.feature_name in f:
                            done = True # for finalization
                            print('Already extracted. Skipped.')
                            return

                patch_count = f['metadata/patch_count'][()]
                batch_idx = [
                    (i, min(i+batch_size, patch_count))
                    for i in range(0, patch_count, batch_size)
                ]

                f.create_dataset(self.feature_name, shape=(patch_count, model.num_features), dtype=np.float32)
                if with_latent_features:
                    # NOTE: using float16 for size efficiencacy
                    f.create_dataset(self.latent_feature_name,
                                     shape=(patch_count, latent_size**2, model.num_features),
                                     dtype=np.float16)

                tq = tqdm_or_st(batch_idx, backend=progress)
                for i0, i1 in tq:
                    coords = f['coordinates'][i0:i1]
                    x = f['patches'][i0:i1]
                    x = (torch.from_numpy(x)/255).permute(0, 3, 1, 2) # BHWC->BCHW
                    x = x.to(self.device)
                    x = (x-mean)/std

                    # with torch.no_grad():
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        h_tensor = model.forward_features(x)

                    h = h_tensor.cpu().detach().numpy() # [B, T+L, H]
                    latent_index = h.shape[1] - latent_size**2
                    # print('latent_index', latent_index)
                    cls_feature, latent_feature = h[:, 0, ...], h[:, latent_index:, ...]

                    f[self.feature_name][i0:i1] = cls_feature
                    if with_latent_features:
                        f[self.latent_feature_name][i0:i1] = latent_feature.astype(np.float16)

                    del x, h_tensor
                    torch.cuda.empty_cache()
                    tq.set_description(f'Processing {i0}-{i1}(total={patch_count})')
                    tq.refresh()

                print('embeddings dimension', f[self.feature_name].shape)
                done = True

            finally:
                if done:
                    print(f'Wrote {self.feature_name}')
                else:
                    del f[self.feature_name]
                    print(f'ABORTED! deleted {self.feature_name}')

                del model, mean, std
                torch.cuda.empty_cache()
                gc.collect()



class ClusterProcessor:
    def __init__(self, hdf5_paths, model_name=DEFAULT_MODEL, cluster_name='', cluster_filter=None):
        assert model_name in ['uni', 'gigapath', 'virchow2']
        self.multi = len(hdf5_paths) > 1
        if self.multi:
            if not cluster_name:
                raise RuntimeError('Multiple files provided but name was not specified.')

        self.hdf5_paths = hdf5_paths
        self.model_name = model_name
        self.cluster_name = cluster_name
        self.sub_clustering = cluster_filter is not None and len(cluster_filter) > 0
        if self.sub_clustering:
            self.cluster_filter = cluster_filter
        else:
            self.cluster_filter = []

        if self.multi:
            self.clusters_path = f'{self.model_name}/clusters_{self.cluster_name}'
        else:
            self.clusters_path = f'{self.model_name}/clusters'

        featuress = []
        clusterss = []

        self.masks = []
        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                patch_count = f['metadata/patch_count'][()]

                # Store if clusters were already calcurated
                if self.clusters_path in f:
                    clusters = f[self.clusters_path][:]
                else:
                    clusters = None

                if len(self.cluster_filter) > 0:
                    if clusters is None:
                        raise RuntimeError('If doing sub-clustering, pre-clustering must be done.')
                    mask = np.isin(clusters, self.cluster_filter)
                else:
                    mask = np.repeat(True, patch_count)
                self.masks.append(mask)

                features = f[f'{self.model_name}/features'][mask]

                if clusters is not None:
                    clusterss.append(clusters[mask])

                featuress.append(features)

        features = np.concatenate(featuress)

        scaler = StandardScaler()
        self.features = scaler.fit_transform(features)

        if len(clusterss) == len(self.hdf5_paths):
            self.has_clusters = True
            self.total_clusters = np.concatenate(clusterss)
        elif len(clusterss) == 0:
            self.has_clusters = False
            self.total_clusters = None
        else:
            raise RuntimeError(f'Count of pre-clustered doesn\'t equal to count of HDF5 files.\n'+
                               f'Pre-cluster count:{len(total_clusterss)} vs HDF5 count:{len(self.hdf5_paths)}')
        self._umap_embeddings = None

    def get_umap_embeddings(self):
        if np.any(self._umap_embeddings):
            return self._umap_embeddings

        reducer = umap.UMAP(
                # n_neighbors=30,
                # min_dist=0.05,
                n_components=2,
                # random_state=a.seed
            )
        embs = reducer.fit_transform(self.features)
        self._umap_embeddings = embs
        return embs


    def anlyze_clusters(self,
                        resolution=1.0,
                        use_umap_embs=False,
                        overwrite=False,
                        progress=DEFAULT_BACKEND
                        ):
        if not self.sub_clustering and self.has_clusters and not overwrite:
            print('Skip clustering')
            return


        self.total_clusters = leiden_cluster(self.features,
                                             umap_emb_func=self.get_umap_embeddings if use_umap_embs else None,
                                             resolution=resolution,
                                             progress=progress)

        # n_clusters = len(set(total_clusters)) - (1 if -1 in total_clusters else 0)
        # n_noise = list(total_clusters).count(-1)

        target_path = self.clusters_path
        if self.sub_clustering:
            suffix = '_sub-' + '-'.join([str(i) for i in self.cluster_filter])
            target_path = target_path + suffix

        print('writing into ', target_path)

        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, self.masks):
            count = np.sum(mask)
            clusters = self.total_clusters[cursor:cursor+count]
            cursor += count
            with h5py.File(hdf5_path, 'a') as f:
                if target_path in f:
                    del f[target_path]
                full_clusters = np.full(len(mask), -1, dtype=clusters.dtype)
                full_clusters[mask] = clusters
                f.create_dataset(target_path, data=full_clusters)

    def plot_umap(self, fig_path=None):
        if not np.any(self.total_clusters):
            raise RuntimeError('Compute clusters before umap projection.')

        fig = plot_umap(embeddings=self.get_umap_embeddings(),
                        clusters=self.total_clusters)

        if fig_path is not None:
            plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.5)
            print(f'wrote {fig_path}')
        return fig


class BasePreviewProcessor:
    def __init__(self, hdf5_path, model_name=DEFAULT_MODEL, size=64):
        self.hdf5_path = hdf5_path
        self.model_name = model_name
        self.size = size

    def load(self, f):
        pass

    def render_patch(self, f, i, patch):
        return patch

    def create_thumbnail(self, progress='tqdm', **kwargs):
        S = self.size
        with h5py.File(self.hdf5_path, 'r') as f:
            self.cols = f['metadata/cols'][()]
            self.rows = f['metadata/rows'][()]
            self.patch_count = f['metadata/patch_count'][()]
            self.patch_size = f['metadata/patch_size'][()]
            self.load(f, **kwargs)

            canvas = Image.new('RGB', (self.cols*S, self.rows*S), (0,0,0))
            tq = tqdm_or_st(range(self.patch_count), backend=progress)
            for i in tq:
                coord = f['coordinates'][i]
                x, y = coord//self.patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch).resize((S, S))
                patch = self.render_patch(f, i, patch)
                canvas.paste(patch, (x, y, x+S, y+S))

        return canvas


class PreviewClustersProcessor(BasePreviewProcessor):
    def load(self, f, **kwargs):
        font_size = kwargs.pop('font_size', 16)
        self.cluster_name = kwargs.pop('cluster_name', '')

        # if None, clusters would be empty
        cluster_path = f'{self.model_name}/clusters'
        if self.cluster_name:
            cluster_path += f'_{self.cluster_name}'
        if cluster_path not in f:
            raise RuntimeError(f'{cluster_path} does not exist in {self.hdf5_path}')
        self.clusters = f[cluster_path][:]

        font = ImageFont.truetype(font=get_platform_font(), size=font_size)

        cmap = plt.get_cmap('tab20')
        self.frames = {}
        for cluster in np.unique(self.clusters).tolist() + [-1]:
            color = mcolors.rgb2hex(cmap(cluster)[:3]) if cluster >= 0 else '#111'
            self.frames[cluster] = create_frame(self.size, color, f'{cluster}', font)

    def render_patch(self, f, i, patch):
        cluster = self.clusters[i]
        if cluster >= 0:
            frame = self.frames[cluster]
            patch.paste(frame, (0, 0), frame)
        return patch



class PreviewScoresProcessor(BasePreviewProcessor):
    def load(self, f, **kwargs):
        font_size = kwargs.pop('font_size', 16)
        score_name = kwargs.pop('score_name', '')

        self.font = ImageFont.truetype(font=get_platform_font(), size=font_size)
        self.scores = f[f'{self.model_name}/scores_{score_name}'][()]
        self.cmap = plt.get_cmap('viridis')

    def render_patch(self, f, i, patch):
        score = self.scores[i]
        if not np.isnan(score):
            color = mcolors.rgb2hex(self.cmap(score)[:3])
            frame = create_frame(self.size, color, f'{score:.3f}', self.font)
            patch.paste(frame, (0, 0), frame)
        return patch



class PreviewLatentPCAProcessor(BasePreviewProcessor):
    def load(self, f, **kwargs):
        alpha = kwargs.pop('alpha', 0.5)
        h = f[f'{self.model_name}/latent_features'][()] # B, L(16x16), EMB(1024)
        h = h.astype(np.float32)
        s = h.shape

        # Estimate original latent size
        latent_size = int(np.sqrt(s[1])) # l = sqrt(L)
        # Validate dyadicity
        assert latent_size**2 == s[1]
        if self.size % latent_size != 0:
            print(f'WARNING: {self.size} is not divident by {latent_size}')

        pca = PCA(n_components=3)
        latent_pca = pca.fit_transform(h.reshape(s[0]*s[1], s[-1])) # B*L, 3

        scaler = MinMaxScaler()
        latent_pca = scaler.fit_transform(latent_pca)

        latent_pca = latent_pca.reshape(s[0], latent_size, latent_size, 3)
        self.overlays = (latent_pca*255).astype(np.uint8) # B, l, l, 3

        self.alpha_mask = Image.new('L', (self.size, self.size), int(alpha*255))


    def render_patch(self, f, i, patch):
        overlay = Image.fromarray(self.overlays[i]).convert('RGBA')
        overlay = overlay.resize((self.size, self.size), Image.NEAREST)
        patch.paste(overlay, (0, 0), self.alpha_mask)
        return patch



class PreviewLatentClusterProcessor(BasePreviewProcessor):
    def load(self, f, **kwargs):
        alpha = kwargs.pop('alpha', 0.5)
        clusters = f[f'{self.model_name}/latent_clusters'][()] # B, L(16x16)
        s = clusters.shape

        # Estimate original latent size
        latent_size = int(np.sqrt(s[1])) # l = sqrt(L)
        # Validate dyadicity
        assert latent_size**2 == s[1]
        if self.size % latent_size != 0:
            print(f'WARNING: {self.size} is not divident by {latent_size}')

        cmap = plt.get_cmap('tab20')
        latent_map = cmap(clusters)
        latent_map = latent_map.reshape(s[0], latent_size, latent_size, 4)
        self.overlays = (latent_map*255).astype(np.uint8) # B, l, l, 4

        self.alpha_mask = Image.new('L', (self.size, self.size), int(alpha*255))


    def render_patch(self, f, i, patch):
        overlay = Image.fromarray(self.overlays[i]).convert('RGBA')
        overlay = overlay.resize((self.size, self.size), Image.NEAREST)
        patch.paste(overlay, (0, 0), self.alpha_mask)
        return patch
