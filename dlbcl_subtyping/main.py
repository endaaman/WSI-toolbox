import os
from tqdm import tqdm
from pydantic import Field
from PIL import Image
import numpy as np
import pandas as pd
import h5py
from openslide import OpenSlide
from gigapath import slide_encoder

from .utils.cli import BaseMLCLI, BaseMLArgs


def is_white_patch(patch, white_threshold=230, white_ratio=0.7):
    gray_patch = np.mean(patch, axis=-1)
    white_pixels = np.sum(gray_patch > white_threshold)
    total_pixels = patch.shape[0] * patch.shape[1]
    return (white_pixels / total_pixels) > white_ratio


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        pass

    # Excute as: uv run main example --param 456
    class ExampleArgs(CommonArgs):
        param: int = 123

    def run_example(self, a):
        print(type(a.param), a.param)
        # m = slide_encoder.create_model()

    class WsiArgs(CommonArgs):
        path: str = 'data/19-0548_2.ndpi'
        patch_size: int = 512
        tile_size: int = 8192
        output_path: str = Field(..., l='--out', s='-o')

    def run_wsi(self, a):
        # m = slide_encoder.create_model()
        S = a.patch_size
        wsi_path = a.path
        wsi = OpenSlide(wsi_path)

        prop = dict(wsi.properties)
        mpp = float(prop['openslide.mpp-x'])
        if 0.360 < mpp < 0.500:
            target_level = 0
            scale = 1
        elif mpp < 0.360:
            target_level = 1
            scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')

        mpp = mpp * scale
        dimension = wsi.level_dimensions[target_level]
        W, H = dimension[0], dimension[1]
        # X = dimension[0] // S
        # Y = dimension[1] // S

        # print('Reading region...')
        # image = wsi.read_region((0, 0), target_level, (512*X, 512*Y))
        # print('region loaded')
        # image = np.array(image.convert('RGB'))
        # print(image.shape)
        # patches = image.reshape(Y, S, X, S, 3)
        # print(patches.shape)
        # patches = patches.transpose(0, 2, 1, 3, 4)  # (Y, X, S, S, 3)
        # print(patches.shape)
        # indices = np.indices((X, Y)).reshape(2, -1).T

        x_patch_count = W//512
        y_patch_count = H//512
        width = (W//512)*512
        row_count = H // 512

        print('target level', target_level)
        print(f'level mpp: {mpp:.6f}')
        print('Original resolutions', W, H)
        print('Obtained resolutions', x_patch_count*S, y_patch_count*S)
        print('row count:', y_patch_count)
        print('col count:', x_patch_count)

        coordinates = []
        total_patches = []
        tq = tqdm(range(row_count))

        with h5py.File(a.output_path, 'w') as f:
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)

            total_patches = f.create_dataset('patches',
                                             shape=(x_patch_count*y_patch_count, S, S, 3),
                                             dtype=np.uint8,
                                             chunks=(1, S, S, 3),
                                             compression='gzip')
            cursor = 0
            for row in tq:
                image = wsi.read_region((0, row*S), target_level, (width, S)).convert('RGB')
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

    class JoinArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field(..., l='--out', s='-o')

    def run_join(self, a):
        S = 32
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
                canvas.paste(patch, (x, y, x+S, y+S))

            canvas.save(a.output_path)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
