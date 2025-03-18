from openslide import OpenSlide
import tifffile
import zarr



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


class WSIProcesser:
    wsi: WSIFile
    def __init__(self, wsi_path, engine='tifffile'):
        if a.engine == 'openslide':
            self.wsi = WSIOpenSlideFile(a.input_path)
        elif a.engine == 'tifffile':
            self.wsi = WSITiffFile(a.input_path)
        else:
            raise ValueError('Invalid engine', a.engine)
        self.original_mpp = self.wsi.get_mpp()
        self.target_level = 0

        if 0.360 < self.original_mpp < 0.500:
            scale = 1
        elif self.original_mpp < 0.360:
            scale = 2
        else:
            raise RuntimeError(f'Invalid scale: mpp={mpp:.6f}')
        self.mpp = self.original_mpp * scale


    def convert_to_hdf5(self, hdf5_path, patch_size=256, progress='tqdm'):
        S = patch_size # scaled patch size
        T = S*self.scale      # actual patch size
        W, H = self.wsi.get_original_size()
        x_patch_count = W//T
        y_patch_count = H//T
        width = (W//T)*T
        row_count = H//T
        coordinates = []
        total_patches = []

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
