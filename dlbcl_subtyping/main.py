import os
import hdbscan
from tqdm import tqdm
from pydantic import Field
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
import seahorse as sns
import h5py
from sklearn.cluster import DBSCAN
from openslide import OpenSlide
import torch
from torchvision import transforms
import timm
from gigapath import slide_encoder
from .utils.cli import BaseMLCLI, BaseMLArgs


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
        pass

    class Wsi2h5Args(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field(..., l='--out', s='-o')
        patch_size: int = 512
        tile_size: int = 8192

    def run_wsi2h5(self, a):
        # m = slide_encoder.create_model()
        S = a.patch_size
        wsi = OpenSlide(a.input_path)

        prop = dict(wsi.properties)
        original_mpp = float(prop['openslide.mpp-x'])
        if 0.360 < original_mpp < 0.500:
            target_level = 0
            scale = 1
        elif original_mpp < 0.360:
            target_level = 1
            scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        mpp = original_mpp * scale

        dimension = wsi.level_dimensions[target_level]
        W, H = dimension[0], dimension[1]

        x_patch_count = W//S
        y_patch_count = H//S
        width = (W//S)*S
        row_count = H // S

        print('target level', target_level)
        print(f'original mpp: {original_mpp:.6f}')
        print(f'image mpp: {mpp:.6f}')
        print('Original resolutions', W, H)
        print('Obtained resolutions', x_patch_count*S, y_patch_count*S)
        print('row count:', y_patch_count)
        print('col count:', x_patch_count)

        coordinates = []
        total_patches = []
        tq = tqdm(range(row_count))


        if os.path.exists(a.output_path):
            if not yes_no_prompt(f'{a.output_path} exists. Overwrite?'):
                print('Aborted.')
                return

        with h5py.File(a.output_path, 'w') as f:
            f.create_dataset('metadata/original_mpp', data=original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=target_level)
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)

            total_patches = f.create_dataset('patches',
                                             shape=(x_patch_count*y_patch_count, S, S, 3),
                                             dtype=np.uint8,
                                             chunks=(1, S, S, 3),
                                             compression='gzip')
            cursor = 0
            for row in tq:
                image = wsi.read_region((0, row*S*scale), target_level, (width, S)).convert('RGB')
                image = np.array(image)

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

    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field(..., l='--out', s='-o')
        size: int = 32
        with_cluster: bool = Field(False, s='-C')

    def run_preview(self, a):
        S = a.size

        cmap = plt.get_cmap('tab10')

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
                    if not 'cluster' in f:
                        raise RuntimeError('cluster data is not detected')
                    cluster = f['cluster'][i]
                    draw = ImageDraw.Draw(patch)
                    if cluster > 0:
                        color = mcolors.rgb2hex(cmap(cluster)[:3])
                    else:
                        color = 'gray'
                    draw.rectangle((0, 0, S, S), outline=color, width=2)
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(a.output_path)

    class ProcessTilesArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        batch_size: int = 32

    def run_process_tiles(self, a):
        tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                         pretrained=True,
                                         dynamic_img_size=True)
        tile_encoder = tile_encoder.eval().to('cuda')

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')

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
                x = x.to('cuda')
                x = (x-mean)/std
                with torch.set_grad_enabled(False):
                    h = tile_encoder(x)
                h = h.cpu().detach()
                hh.append(h)
            hh = torch.cat(hh).numpy()

        print(hh.shape)

        assert len(hh) == patch_count

        with h5py.File(a.input_path, 'w') as f:
            f.create_dataset('features', data=hh)


    class ClusterArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')

    def run_cluster(self, a):
        with h5py.File(a.input_path, 'a') as f:
            features = f['features']

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=a.seed)
        embedding = reducer.fit_transform(scaled_features)

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(embedding)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print('n_clusters', n_clusters)
        print('n_noise', n_noise)


        plt.figure(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(set(clusters))))
        for cluster_id, color in zip(set(clusters), colors):
            if cluster_id == -1:
                cluster_points = embedding[clusters == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, c='black', label='Noise')
            else:
                cluster_points = embedding[clusters == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, c=[color], label=f'Cluster {cluster_id}')

        plt.title('UMAP + HDBSCAN Clustering')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    cli = CLI()
    cli.run()
