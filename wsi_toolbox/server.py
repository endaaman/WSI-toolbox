import streamlit as st
import os
import pandas as pd
from pathlib import Path
import time

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

    # デフォルトルートと現在のディレクトリの絶対パス取得
    default_root_abs = os.path.abspath(DEFAULT_ROOT)
    current_dir_abs = os.path.abspath(st.session_state.current_dir)

    # 親ディレクトリボタン処理
    if current_dir_abs != default_root_abs:
        if st.button('↑ 親フォルダへ'):
            # 親ディレクトリを取得
            parent_dir = os.path.dirname(current_dir_abs)
            # デフォルトルート以上に移動しないようチェック
            if os.path.commonpath([default_root_abs]) == os.path.commonpath([default_root_abs, parent_dir]):
                st.session_state.current_dir = parent_dir
                st.rerun()
    else:
        # ルートディレクトリの場合は無効化
        st.button('↑ 親フォルダへ', disabled=True)


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

        cluster_name = None
        if multi and operation_index == 0:
            cluster_name = st.text_input('複数WSIを同時にクラスタリングする場合はクラスタ名を入力してください。', key='wsi_cluster_name')

        if st.button('処理を実行', key='process_wsi'):
            if multi and (not cluster_name):
                st.error('複数同時処理の場合はクラスタ名を入力してください。')
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process each file
                h5_paths = []
                for i, file in enumerate(selected_files):
                    # Update progress
                    progress = int((i / len(selected_files)) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f'Processing {file["name"]}...')

                    # Convert to h5
                    h5_path = convert_wsi_to_h5(file['path'])

                    # Extract features
                    h5_path = extract_features(h5_path)
                    h5_paths.append(h5_path)

                # Perform clustering if selected
                if 'Clustering' in operation:
                    perform_clustering(h5_paths, cluster_name)

                # Complete progress
                progress_bar.progress(100)
                status_text.text('Processing completed!')
    elif mode == 'HDF5':
        st.subheader('Processing Options')
        st.write('HDF5 files selected. Available operations:')

        # For multiple files, need a cluster name
        cluster_name = None
        if multi:
            cluster_name = st.text_input('Cluster name (required for multiple files)', key='h5_cluster_name')

        if st.button('Perform Clustering', key='process_h5'):
            if len(selected_files) > 1 and (not cluster_name or cluster_name.strip() == ''):
                st.error('Cluster name is required for multiple files')
            else:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process each file
                h5_paths = [file['path'] for file in selected_files]

                # Check if features exist, otherwise extract them
                status_text.text('Checking for features...')

                # Mock checking for features (in reality would check if features exist in h5)
                for i, h5_path in enumerate(h5_paths):
                    progress = int((i / len(h5_paths)) * 50)
                    progress_bar.progress(progress)
                    status_text.text(f'Extracting features from {os.path.basename(h5_path)}...')
                    extract_features(h5_path)

                # Perform clustering
                status_text.text('Performing clustering...')
                progress_bar.progress(50)
                perform_clustering(h5_paths, cluster_name)

                # Complete progress
                progress_bar.progress(100)
                status_text.text('Processing completed!')
    else:
        st.warning(f'Invalid mode: {mode}')

if __name__ == '__main__':
    main()
