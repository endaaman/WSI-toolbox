import re
import time
import os
from pathlib import Path
import time
import sys
import warnings

import numpy as np
from PIL import Image
import h5py
import torch
import pandas as pd

torch.classes.__path__ = []
import streamlit as st

sys.path.append(str(Path(__file__).parent))
__package__ = 'wsi_toolbox'

from .utils.progress import tqdm_or_st
from .processor import WSIProcessor, TileProcessor, ClusterProcessor, ThumbProcessor


Image.MAX_IMAGE_PIXELS = 3_500_000_000
warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

st.set_page_config(
    page_title='WSI Analysis System',
    page_icon='🔬',
    layout='wide'
)


def is_wsi_file(file_path):
    extensions = ['.ndpi', '.svs']
    return Path(file_path).suffix.lower() in extensions

def is_h5_file(file_path):
    return Path(file_path).suffix.lower() == '.h5'

def get_hdf5_detail(hdf_path):
    with h5py.File(hdf_path, 'r') as f:
        if 'metadata/patch_count' not in f:
            return {
                'supported': False,
                'has_features': False,
                'cluster_names': ['未施行'],
                'patch_count': 0,
                'mpp': 0,
                'cols': 0,
                'rows': 0,
            }
        patch_count = f['metadata/patch_count'][()]
        has_features = 'gigapath/features' in f and (len(f['gigapath/features']) == patch_count)
        cluster_names = ['未施行']
        if 'gigapath' in f:
            cluster_names = [
                k.replace('clusters_', '').replace('clusters', 'デフォルト')
                for k in f['gigapath'].keys() if re.match(r'^clusters.*', k)
            ]
        return {
            'supported': True,
            'has_features': has_features,
            'cluster_names': cluster_names,
            'patch_count': patch_count,
            'mpp': f['metadata/mpp'][()],
            'cols': f['metadata/cols'][()],
            'rows': f['metadata/rows'][()],
        }


IMAGE_EXTENSIONS = { '.bmp', '.gif', '.icns', '.ico', '.jpg', '.jpeg', '.png', '.tif', '.tiff', }
def is_image_file(file_path):
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def list_files(directory):
    files = []
    directories = []

    for item in sorted(os.listdir(directory)):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            icon = "📄"
            file_type = "Other"
            detail = None
            if is_wsi_file(item_path):
                icon = '🔬'
                file_type = "WSI"
            elif is_h5_file(item_path):
                icon = '📊'
                file_type = "HDF5"
                detail = get_hdf5_detail(item_path)
            elif is_image_file(item_path):
                icon = '🖼️'
                file_type = "Image"

            size = os.path.getsize(item_path)
            if size > 1024*1024*1024:
                size_str = f'{size/1024/1024/1024:.1f} GB'
            elif size > 1024*1024:
                size_str = f'{size/1024/1024:.1f} MB'
            elif size > 1024:
                size_str = f'{size/1024:.1f} KB'
            else:
                size_str = f'{size} bytes'

            files.append({
                "selected": False,
                "name": f'{item} {icon}',
                "path": item_path,
                "type": file_type,
                "size": size_str,
                "modified": time.ctime(os.path.getmtime(item_path)),
                "detail": detail,
            })

        elif os.path.isdir(item_path):
            directories.append({
                "selected": False,
                "name": f"📁 {item}",
                "path": item_path,
                "type": "Directory",
                "size": "",
                "modified": time.ctime(os.path.getmtime(item_path)),
                "detail": None,
            })

    all_items = directories + files
    return all_items



def get_mode_and_multi(selected_files):
    if len(selected_files) == 0:
        return 'Empty', False
    if len(selected_files) == 1:
        selected = selected_files[0]
        return selected['type'], False

    type_set = set([f['type'] for f in selected_files])
    if len(type_set) > 1:
        return 'Mix', True
    t = next(iter(type_set))
    return t, True


def format_size(size_bytes):
    return 'AA'
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f} GB"


BASE_DIR = os.getenv('BASE_DIR', 'data')

def main():
    st.title('WSI AI解析システム')

    if 'current_dir' not in st.session_state:
        st.session_state.current_dir = BASE_DIR

    default_root_abs = os.path.abspath(BASE_DIR)
    current_dir_abs = os.path.abspath(st.session_state.current_dir)

    cols = st.columns([0.1, 0.1])
    with cols[0]:
        if current_dir_abs == default_root_abs:
            st.button('↑ 親フォルダへ', disabled=True)
        else:
            if st.button('↑ 親フォルダへ'):
                parent_dir = os.path.dirname(current_dir_abs)
                if os.path.commonpath([default_root_abs]) == os.path.commonpath([default_root_abs, parent_dir]):
                    st.session_state.current_dir = parent_dir
                    st.rerun()

    with cols[1]:
        if st.button('フォルダ更新'):
            st.rerun()


    files = list_files(st.session_state.current_dir)
    files_df = pd.DataFrame(files)

    st.subheader(f'ファイル選択')

    if len(files_df) ==  0:
        st.warning('ファイルが選択されていません')
        return

    edited_df = st.data_editor(
        files_df,
        column_config={
            'selected': st.column_config.CheckboxColumn(
                label='✓',
                width='small',
                # help='Select files'
            ),
            'name': 'ファイル名',
            'type': '種別',
            'size': 'ファイルサイズ',
            'modified': st.column_config.DateColumn(
                'Birthday',
                format='YYYY/MM/DD hh:mm:ss',
            ),
            'path': None,  # Hide path column
            'detail': None,  # Hide path column
        },
        hide_index=True,
        use_container_width=True,
        disabled=['name', 'type', 'size', 'modified'],
    )

    # selected_files = edited_df[edited_df['selected'] == True].to_dict('records')
    selected_indices = edited_df[edited_df['selected'] == True].index.tolist()
    selected_files = [files[i] for i in selected_indices]

    mode, multi = get_mode_and_multi(selected_files)
    if mode == 'Empty':
        st.write('チェックボックからファイルを選択してください。')
    elif mode == 'Directory':
        if multi:
            st.warning('複数フォルダが選択されました。')
        else:
            if st.button('このフォルダに移動'):
                st.session_state.current_dir = selected_files[0]['path']
                st.rerun()

    elif mode == 'Other':
        st.warning('WSI(.ndpi, .svs)ファイルもしくはHDF5ファイル(.h5)を選択しください。')
    elif mode == 'Mix':
        st.warning('単一種類のファイルを選択してください。')
    elif mode == 'Image':
        for f in selected_files:
            img = Image.open(f['path'])
            st.image(img)
    elif mode == 'WSI':
        st.subheader('HDF5に変換し特徴量を抽出する')
        st.write('変換と特徴量抽出の2ステップを実行します。どちらも結構時間がかかります。')

        do_clustering = st.checkbox(
            'クラスタリングも実行する(クラスタリングも同時に行うには一つだけ選択してください。)',
            disabled=multi, value=not multi)

        if st.button('処理を実行', key='process_wsi'):
            wsi_path = selected_files[0]['path']
            basename = os.path.splitext(wsi_path)[0]
            hdf5_path = f'{basename}.h5'
            hdf5_tmp_path = f'{basename}.h5.tmp'
            wp = WSIProcessor(wsi_path)
            with st.spinner('HDF5ファイルに変換中...', show_time=True):
                wp.convert_to_hdf5(hdf5_tmp_path, patch_size=256, progress='streamlit')
            os.rename(hdf5_tmp_path, hdf5_path)
            st.write('HDF5ファイルに変換完了。')

            tp = TileProcessor(model_name='gigapath', device='cuda')
            with st.spinner('特徴量抽出中...', show_time=True):
                tp.evaluate_hdf5_file(hdf5_path, batch_size=256, progress='streamlit', overwrite=True)
            st.write('特徴量抽出完了。')

            if multi:
                st.write('すべての処理が完了しました。')
                st.write('※クラスタリングも実行する場合はHDF5から選択してください。')
            else:
                if do_clustering:
                    cluster_proc = ClusterProcessor(
                            selected_files[0]['path'],
                            model_name='gigapath',
                            cluster_name='')
                    resolution = 1.0
                    # resolution = st.slider('クラスタリング解像度',
                    #                        min_value=0.0, max_value=3.0,
                    #                        value=1.0, step=0.1)
                    with st.spinner(f'クラスタリング中...', show_time=True):
                        cluster_proc.anlyze_clusters(resolution)
                    st.write('クラスタリング完了。')
                    base, ext = os.path.splitext(selected_files[0]['path'])
                    umap_path = f'{base}_umap.png'
                    cluster_proc.save_umap(umap_path)
                    img = Image.open(umap_path)
                    st.image(img)
                st.write('すべての処理が完了しました。')

    elif mode == 'HDF5':
        st.subheader('HDF5ファイル解析オプション')
        df_details = pd.DataFrame([{'name': f['name'], **f['detail']} for f in selected_files])
        if not np.all(df_details['supported']):
            st.error('サポートされていないHDF5ファイルが選択されました。')
        else:
            df_details['has_features'] = df_details['has_features'].map({True: '抽出済み', False: '未抽出'})
            st.dataframe(
                df_details,
                column_config={
                    'name': 'ファイル名',
                    'has_features': '特徴量抽出状況',
                    'cluster_names': 'クラスタリング処理状況',
                    'patch_count': 'パッチ数',
                    'mpp': 'micro/pixel',
                    'supported': None,
                },
                hide_index=True,
                use_container_width=False,
            )

            ok = True
            cluster_name = ''
            if multi:
                cluster_name = st.text_input(
                    '',
                    value='',
                    placeholder='半角英数字でクラスタ名を入力してください',
                )
                if not re.match(r'[a-zA-Z0-9_-]+', cluster_name):
                    st.error('複数同時処理の場合はクラスタ名を入力してください。')
                    ok = False

            resolution = 1.0
            # resolution = st.slider('クラスタリング解像度',
            #                        min_value=0.0, max_value=3.0,
            #                        value=1.0, step=0.1)
            # overwrite = False
            overwrite = st.checkbox('計算済みクラスタ結果を再度計算する', value=False)

            if ok and st.button('クラスタリングを実行', key='process_wsi'):
                for f in selected_files:
                    if not f['detail']['has_features']:
                        st.write(f'{f["name"]}の特徴量が未抽出なので、抽出を行います。')
                        st.write(f'特徴量抽出中...')
                        tile_proc = TileProcessor(model_name='gigapath', device='cuda')
                        tile_proc.evaluate_hdf5_file(f['path'], batch_size=256, progress='streamlit', overwrite=True)
                        st.write('特徴量抽出完了。')

                cluster_proc = ClusterProcessor(
                        [f['path'] for f in selected_files],
                        model_name='gigapath',
                        cluster_name=cluster_name)
                t = 'と'.join([f['name'] for f in selected_files])
                with st.spinner(f'{t}をクラスタリング中...', show_time=True):
                    if multi:
                        dir = os.path.dirname(selected_files[0]['path'])
                        umap_path = f'{dir}/{cluster_name}.png'
                    else:
                        base, ext = os.path.splitext(selected_files[0]['path'])
                        umap_path = f'{base}_umap.png'
                    cluster_proc.anlyze_clusters(resolution=resolution, overwrite=overwrite,
                                                 use_umap_embs=False, progress='streamlit')
                    cluster_proc.save_umap(umap_path)

                st.write('クラスタリング完了。')
                st.image(Image.open(umap_path))

                with st.spinner('サムネイル生成中', show_time=True):
                    for file in selected_files:
                        thumb_proc = ThumbProcessor(file['path'], cluster_name=cluster_name, size=64)
                        base, ext = os.path.splitext(file['path'])
                        if multi:
                            thumb_path = f'{base}_thumb_{cluster_name}.jpg'
                        else:
                            thumb_path = f'{base}_thumb.jpg'
                        thumb_proc.create_thumbnail(thumb_path, progress='streamlit')
                        st.write(thumb_path)
                        st.image(Image.open(thumb_path))
                    st.write('サムネイル生成完了')

    else:
        st.warning(f'Invalid mode: {mode}')

if __name__ == '__main__':
    main()
