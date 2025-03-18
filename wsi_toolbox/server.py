import re
import time
import os
from pathlib import Path
import time
import sys
import warnings

import numpy as np
import h5py
import torch
import streamlit as st
torch.classes.__path__ = []
import pandas as pd

sys.path.append(str(Path(__file__).parent))
__package__ = 'wsi_toolbox'

from .utils.progress import tqdm_or_st
from .processor import WSIProcessor, TileProcessor

warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

st.set_page_config(
    page_title='WSI Analysis System',
    page_icon='🔬',
    layout='wide'
)


def is_wsi_file(file_path):
    '''Check if file is a WSI file based on extension'''
    extensions = ['.ndpi', '.svs']
    return Path(file_path).suffix.lower() in extensions

def is_h5_file(file_path):
    '''Check if file is an HDF5 file'''
    return Path(file_path).suffix.lower() == '.h5'

def get_hdf5_detail(hdf_path):
    with h5py.File(hdf_path, 'r') as f:
        if 'metadata/patch_count' not in f:
            return {
                'supported': False,
                'has_features': False,
                'patch_count': 0,
                'mpp': 0,
                'cols': 0,
                'rows': 0,
            }
        patch_count = f['metadata/patch_count'][()]
        has_features = 'gigapath/features' in f and (len(f['gigapath/features']) == patch_count)
        return {
            'supported': True,
            'has_features': has_features,
            'patch_count': patch_count,
            'mpp': f['metadata/mpp'][()],
            'cols': f['metadata/cols'][()],
            'rows': f['metadata/rows'][()],
        }

def list_files(directory):
    files = []
    directories = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            icon = "📄"
            file_type = "Other"
            detail = None
            if is_wsi_file(item_path):
                icon = "🔬"
                file_type = "WSI"
            elif is_h5_file(item_path):
                icon = "📊"
                file_type = "HDF5"
                detail = get_hdf5_detail(item_path)

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
                "name": f"{icon} {item}",
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




DEFAULT_ROOT = 'data'

def main():
    st.title('WSI AI解析システム')

    if 'current_dir' not in st.session_state:
        st.session_state.current_dir = DEFAULT_ROOT

    default_root_abs = os.path.abspath(DEFAULT_ROOT)
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
            'modified': 'Last Modified',
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
    elif mode == 'WSI':
        st.subheader('HDF5に変換し特徴量を抽出する')
        st.write('変換と特徴量抽出の2ステップを実行します。結構時間がかかります。')

        do_clustering = st.checkbox(
            'クラスタリングも実行する' + '' if not multi else '(クラスタリングも同時に行うには単数選択してください。)',
            disabled=multi, value=not multi)

        if st.button('処理を実行', key='process_wsi'):
            st.write('HDF5ファイルに変換中...')
            wsi_path = selected_files[0]['path']
            basename = os.path.splitext(wsi_path)[0]
            hdf5_path = f'{basename}.h5'
            hdf5_tmp_path = f'{basename}.h5.tmp'
            wp = WSIProcessor(wsi_path)
            wp.convert_to_hdf5(hdf5_tmp_path, patch_size=256, progress='streamlit')
            os.rename(hdf5_tmp_path, hdf5_path)
            st.write('HDF5ファイルに変換完了。')

            st.write('による特徴量抽出中...')
            tp = TileProcessor(model_name='gigapath', device='cuda')
            tp.evaluate_hdf5_file(hdf5_path, batch_size=256, progress='streamlit', overwrite=True)
            st.write('特徴量抽出完了。')

            if multi:
                st.write('すべての処理が完了しました。')
                st.write('クラスタリングも実行する場合はHDF5から選択してください。')
            else:
                st.write('クラスタリング中...')
                time.sleep(1)
                st.write('クラスタリング完了。')


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
                    'patch_count': 'パッチ数',
                    'mpp': 'micro/pixel',
                    'supported': None,
                },
                hide_index=True,
                use_container_width=False,
            )

            if st.button('クラスタリングを実行', key='process_wsi'):
                for f in selected_files:
                    if not f['detail']['has_features']:
                        st.write(f'{f["name"]}の特徴量抽出中...')
                        tp = TileProcessor(model_name='gigapath', device='cuda')
                        tp.evaluate_hdf5_file(f['path'], batch_size=256, progress='streamlit', overwrite=True)
                        st.write('特徴量抽出完了。')

                st.write('クラスタリング中...')
                time.sleep(1)
                st.write('クラスタリング完了。')


        # if multi and not cluster_name:
        #     st.error('複数同時処理の場合はクラスタ名を入力してください。')
        #     ok = False
        # elif multi and not re.match(r'[a-zA-Z0-9_-]+', cluster_name):
        #     st.error('半角英数のみで入力してください。')
        #     ok = False

        # ok = True
        # if st.button('クラスタリングを実行', key='process_wsi'):
        #     if ok:

        #         if operation_index == 0:
        #             # HDF5変換のみ
        #             pass
        #         elif operation_index > 0:
        #             # 特徴量抽出
        #             tp = TileProcessor(model_name='gigapath', device='cuda')
        #             tp.evaluate_hdf5_file(hdf5_path, progress='streamlit')

        #             if operation_index > 1:
        #                 # HDF5変換+特徴量抽出+クラスタリング
        #                 pass
        #         st.write('処理完了')

    else:
        st.warning(f'Invalid mode: {mode}')

if __name__ == '__main__':
    main()
