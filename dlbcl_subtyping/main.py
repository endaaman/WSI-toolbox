import os
from tqdm import tqdm
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

        x_patch_count = W//S
        y_patch_count = H//S
        x_tile_count = W//a.tile_size
        y_tile_count = H//a.tile_size

        tile_widths = np.array([(x_patch_count + i) // x_tile_count for i in range(x_tile_count)])*S
        tile_heights = np.array([(y_patch_count + i) // y_tile_count for i in range(y_tile_count)])*S

        print('Original size', W, H)
        print('read size', x_patch_count*512, y_patch_count*512)
        print('tile count', x_tile_count, y_tile_count)
        print('patch count', x_patch_count, y_patch_count)

        output_path = 'out/a.h5'
        with h5py.File(output_path, 'w') as f:
            images = f.create_dataset('patches',
                                     shape=(x_patch_count*y_patch_count, S, S, 3),
                                     dtype='uint8',
                                     chunks=True,
                                     compression='gzip')
            patch_index = 0
            coordinates = []
            tile_x = 0
            tile_y = 0
            cursor = 0
            for h in tile_heights:
                tile_x = 0
                for w in tile_widths:
                    print((tile_x, tile_y), w, h, )
                    tile = wsi.read_region((tile_x, tile_y), target_level, (w, h)).convert('RGB')
                    print(f'{tile_y} {tile_x}', w, h)
                    # tile.save(f'out/{tile_y}_{tile_x}.jpg')
                    os.makedirs(f'out/{tile_y}_{tile_x}', exist_ok=True)

                    tile = np.array(tile)
                    cols = w//S
                    rows = h//S
                    patches = tile.reshape(rows, S, cols, S, 3) # (y, h, x, w, 3)
                    patches = patches.transpose(0, 2, 1, 3, 4)   # (y, x, h, w, 3)
                    indices = np.indices((cols, rows)).reshape(2, -1).T
                    batch = []
                    for (x, y) in tqdm(indices):
                        patch = patches[y, x, ...]
                        # Image.fromarray(patch).save(f'out/{tile_y}_{tile_x}/{y}_{x}.jpg')
                        if is_white_patch(patch):
                            continue
                        coordinates.append((x, y))
                        batch.append(patch)
                        patch_index += 1
                    batch = np.array(batch)
                    print('batch', batch.shape)
                    images[cursor:len(batch), ...] = batch
                    cursor += len(batch)

                    tile_x += w

                tile_y += h

            # for x, y in tqdm(indices):
            #     img = wsi.read_region((x*S, y*S), target_level, (512, 512))
            #     patch = np.array(img.convert('RGB'))
            #     if is_white_patch(patch):
            #         continue
            #     images[i] = patch
            #     i += 1
            #     coordinates.append((x*S, y*S))

            f.create_dataset('coordinates', data=coordinates)

        print('done')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
