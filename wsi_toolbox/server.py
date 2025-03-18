import re
import os
from pathlib import Path
import time
import sys

import torch
import streamlit as st
torch.classes.__path__ = []
import pandas as pd

sys.path.append(str(Path(__file__).parent))
__package__ = 'wsi_toolbox'

from .utils.progress import tqdm_or_st
from .wsi import WSIProcesser

# Set page config
st.set_page_config(
    page_title='WSI Analysis System',
    page_icon='🔬',
    layout='wide'
)

# Mock functions for processing steps (to be implemented separately)
def convert_wsi_to_h5(file_path, output_dir=None):
    '''Mock function to convert WSI to HDF5 with patches'''
    st.info(f'Converting {file_path} to h5...')
    # Simulate processing time
    time.sleep(2)
    output_path = str(file_path).replace(Path(file_path).suffix, '.h5')
    st.success(f'Converted to {output_path}')
    return output_path

def extract_features(h5_path):
    '''Mock function to extract features using Prov-GigaPath model'''
    st.info(f'Extracting features from {h5_path}...')
    # Simulate processing time
    time.sleep(3)
    st.success(f'Features extracted and saved to {h5_path}')
    return h5_path

def perform_clustering(h5_paths, name=None):
    '''Mock function for clustering features using leidenalg'''
    st.info(f'Performing clustering on {len(h5_paths)} files...')
    # Simulate processing time
    time.sleep(4)

    suffix = f'_{name}' if name else ''
    output_files = []

    for h5_path in h5_paths:
        base_path = str(h5_path).replace('.h5', '')
        umap_path = f'{base_path}{suffix}_umap.jpg'
        preview_path = f'{base_path}{suffix}_preview.jpg'
        output_files.append((umap_path, preview_path))

    st.success(f'Clustering completed. UMAP and preview images generated.')
    return output_files

def is_wsi_file(file_path):
    '''Check if file is a WSI file based on extension'''
    extensions = ['.ndpi', '.svs']
    return Path(file_path).suffix.lower() in extensions

def is_h5_file(file_path):
    '''Check if file is an HDF5 file'''
    return Path(file_path).suffix.lower() == '.h5'

def list_files(directory):
    files = []
    directories = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            icon = "📄"
            file_type = "Other"
            if is_wsi_file(item_path):
                icon = "🔬"
                file_type = "WSI"
            elif is_h5_file(item_path):
                icon = "📊"
                file_type = "HDF5"

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
                "modified": time.ctime(os.path.getmtime(item_path))
            })

        elif os.path.isdir(item_path):
            directories.append({
                "selected": False,
                "name": f"📁 {item}",
                "path": item_path,
                "type": "Directory",
                "size": "",
                "modified": time.ctime(os.path.getmtime(item_path))
            })

    # ディレクトリを先に、次にファイルを表示
    all_items = directories + files
    return pd.DataFrame(all_items)


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
        },
        hide_index=True,
        use_container_width=True,
        disabled=['name', 'type', 'size', 'modified'],
    )

    selected_files = edited_df[edited_df['selected'] == True].to_dict('records')

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
        st.subheader('WSI解析オプション')

        operations = [
            'HDF5に変換+特徴量抽出+クラスタリング',
            'HDF5に変換+特徴量抽出',
            'HDF5に変換のみ',
        ]
        operation = st.radio('処理の内容', operations, index=0)
        operation_index = operations.index(operation)

        st.write(f'{multi} {operation_index}')

        # cluster_name = None
        # if multi and operation_index == 0:
        #     cluster_name = st.text_input('複数WSIを同時にクラスタリングする場合はクラスタ名を入力してください。', key='wsi_cluster_name')

        ok = True
        if st.button('処理を実行', key='process_wsi'):
            if multi:
                st.error('一つだけ選択してください。')
                ok = False
            # if multi and not cluster_name:
            #     st.error('複数同時処理の場合はクラスタ名を入力してください。')
            #     ok = False
            # elif multi and not re.match(r'[a-zA-Z0-9_-]+', cluster_name):
            #     st.error('半角英数のみで入力してください。')
            #     ok = False

            if ok:
                st.write('処理開始')
                input_path = selected_files[0]['path']
                output_path = f'{os.path.splitext(input_path)[0]}.h5'
                p = WSIProcesser(input_path)
                p.convert_to_hdf5(output_path, patch_size=256, progress='streamlit')


    elif mode == 'HDF5':
        st.subheader('HDF5ファイル解析オプション')

        valid = True
        operations = [
            '特徴量抽出+クラスタリング',
            '特徴量抽出',
        ]
        operation = st.radio('処理の内容', operations, index=0)
        operation_index = operations.index(operation)

    else:
        st.warning(f'Invalid mode: {mode}')

if __name__ == '__main__':
    main()
