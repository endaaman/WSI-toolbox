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
import seahorse as sns
import h5py
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from openslide import OpenSlide
import torch
from torchvision import transforms
from torch.amp import autocast
import timm
from gigapath import slide_encoder

from .utils.cli import BaseMLCLI, BaseMLArgs


warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


def yes_no_prompt(question):
    print(f"{question} [Y/n]: ", end="")
    response = input().lower()
    return response == "" or response.startswith("y")

def is_white_patch(patch, white_threshold=230, white_ratio=0.7):
    gray_patch = np.mean(patch, axis=-1)
    white_pixels = np.sum(gray_patch > white_threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    return (white_pixels / total_pixels) > white_ratio


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass

    class Wsi2h5Args(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field(..., l='--out', s='-o')
        patch_size: int = 256
        tile_size: int = 8192
        overwrite: bool = False

    def run_wsi2h5(self, a):
        if os.path.exists(a.output_path):
            if not a.overwrite:
                print(f'{a.output_path} exists. Skipping.')
                return
            print(f'{a.output_path} exists but overwriting it.')

        wsi = OpenSlide(a.input_path)

        prop = dict(wsi.properties)
        original_mpp = float(prop['openslide.mpp-x'])
        if 0.360 < original_mpp < 0.500:
            target_level = 0
            scale = 1
        elif original_mpp < 0.360:
            target_level = 0
            scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        mpp = original_mpp * scale

        S = a.patch_size # scaled patch size
        T = S*scale      # actual patch size

        dimension = wsi.level_dimensions[target_level]
        W, H = dimension[0], dimension[1]

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
        tq = tqdm(range(row_count))

        os.makedirs(os.path.dirname(a.output_path), exist_ok=True)

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

            total_patches = f.create_dataset('patches',
                                             shape=(x_patch_count*y_patch_count, S, S, 3),
                                             dtype=np.uint8,
                                             chunks=(1, S, S, 3),
                                             compression='gzip',
                                             compression_opts=9
                                             )
            cursor = 0
            for row in tq:
                image = wsi.read_region((0, row*T), target_level, (width, T)).convert('RGB')
                image = np.array(image)
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

            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        print(f'{len(coordinates)} patches were selected.')
        print('done')

    class ThumbnailArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        size: int = 64
        with_cluster: bool = Field(False, s='-C')
        show: bool = False

    def run_thumbnail(self, a):
        S = a.size

        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}.jpg'

        cmap = plt.get_cmap('tab20')

        with h5py.File(a.input_path, 'r') as f:
            cols = f['metadata/cols'][()]
            rows = f['metadata/rows'][()]
            patch_count = f['metadata/patch_count'][()]
            patch_size = f['metadata/patch_size'][()]

            canvas = Image.new('RGB', (cols*S, rows*S), (0,0,0))
            for i in tqdm(range(patch_count)):
                coord = f['coordinates'][i]
                x, y = coord//patch_size*S
                patch = f['patches'][i]
                patch = Image.fromarray(patch)
                patch = patch.resize((S, S))
                if a.with_cluster:
                    if not 'clusters' in f:
                        raise RuntimeError('clusters data is not found')
                    cluster = f['clusters'][i]
                    draw = ImageDraw.Draw(patch)
                    if cluster > 0:
                        color = mcolors.rgb2hex(cmap(cluster)[:3])
                    else:
                        color = 'gray'
                    draw.rectangle((0, 0, S, S), outline=color, width=4)
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(output_path)
            print(f'wrote {output_path}')

        if a.show:
            os.system(f'xdg-open {output_path}')

    class ProcessPatchesArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        batch_size: int = Field(32, s='-B')
        overwrite: bool = False

    def run_process_patches(self, a):
        with h5py.File(a.input_path, 'r') as f:
            if 'features' in f:
                if not a.overwrite:
                    print('patch embeddings are already obtained.')
                    return

        tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                         pretrained=True,
                                         dynamic_img_size=True)
        tile_encoder = tile_encoder.eval().to(a.device)

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
                    h = tile_encoder(x)
                h = h.cpu().detach()
                hh.append(h)
            hh = torch.cat(hh).numpy()

        print('embeddings dimension', hh.shape)
        assert len(hh) == patch_count

        with h5py.File(a.input_path, 'a') as f:
            if a.overwrite and 'features' in f:
                print('Overwriting features.')
                del f['features']
            f.create_dataset('features', data=hh)


    class ProcessSlideArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        overwrite: bool = False

    def run_process_slide(self, a):
        with h5py.File(a.input_path, 'r') as f:
            if 'slide_feature' in f:
                if not a.overwrite:
                    print('feature embeddings are already obtained.')
                    return

            features = f['features'][:]
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
        method: str = Field('HDBSCAN', s='-M')
        save: bool = False
        fig_path: str = ''
        noshow: bool = False

    def run_cluster(self, a):
        with h5py.File(a.input_path, 'r') as f:
            features = f['features'][:]
        print('Loaded features', features.shape)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        print('UMAP fitting...')
        reducer = umap.UMAP(
                n_neighbors=30,
                min_dist=0.1,
                n_components=2,
                # random_state=a.seed
            )
        embedding = reducer.fit_transform(scaled_features)
        print('Loaded features', features.shape)

        eps = 0.2
        if a.method.lower() == 'dbscan':
            m = DBSCAN(
                eps=eps,
                min_samples=5
            )
            clusters = m.fit_predict(embedding)
        elif a.method.lower() == 'hdbscan':
            m = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=5,
                cluster_selection_epsilon=eps,
                metric='euclidean'
            )
            clusters = m.fit_predict(embedding)
        elif a.method.lower() == 'snn':
            k = 30  # 近傍数
            nn = NearestNeighbors(n_neighbors=k).fit(embedding)
            distances, indices = nn.kneighbors(embedding)
            n_samples = embedding.shape[0]
            snn_graph = np.zeros((n_samples, n_samples))

            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    # i と j の共有近傍の数を計算
                    shared_neighbors = len(set(indices[i]) & set(indices[j]))
                    if shared_neighbors > 0:
                        similarity = shared_neighbors / k
                        snn_graph[i, j] = similarity
                        snn_graph[j, i] = similarity

            m = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='precomputed'
            )
            # 距離行列に変換（類似度が高いほど距離が小さい）
            distance_matrix = 1 - snn_graph
            clusters = m.fit_predict(distance_matrix)
        else:
            raise RuntimeError('Invalid medthod:', a.method)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print('n_clusters', n_clusters)
        print('n_noise', n_noise)


        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab20')
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(set(clusters))))
        cluster_ids = sorted(list(set(clusters)))
        for i, cluster_id in enumerate(cluster_ids):
            coords = embedding[clusters == cluster_id]
            if cluster_id == -1:
                color = 'black'
                label = 'Noise'
                size = 20
            else:
                color = [cmap(cluster_id % 20)]
                label = f'Cluster {cluster_id}'
                size = 10
            plt.scatter(coords[:, 0], coords[:, 1], s=size, c=color, label=label)

        plt.title(f'UMAP + {a.method} Clustering')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if a.save:
            with h5py.File(a.input_path, 'a') as f:
                if 'clusters' in f:
                    del f['clusters']
                f.create_dataset('clusters', data=clusters)
            print(f'Save clusters to {a.input_path}')

            fig_path = a.fig_path
            if not fig_path:
                base, ext = os.path.splitext(a.input_path)
                fig_path = f'{base}_umap.png'
            plt.savefig(fig_path)
            print(f'wrote {fig_path}')

        if not a.noshow:
            plt.show()

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


if __name__ == '__main__':
    cli = CLI()
    cli.run()
