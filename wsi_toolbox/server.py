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

            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            files.append({
                "selected": False,
                "name": f"{icon} {item}",
                "path": item_path,
                "type": file_type,
                "size_mb": f"{size_mb:.2f} MB",
                "modified": time.ctime(os.path.getmtime(item_path))
            })

        elif os.path.isdir(item_path):
            # ディレクトリ用アイコン
            directories.append({
                "selected": False,
                "name": f"📁 {item}",
                "path": item_path,
                "type": "Directory",
                "size_mb": "",
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
        return [selected['type'], False

    type_set = set([f['type'] for f in selected_files])
    if len(type_set) > 1:
        return 'Mix', True
    st.write(type_set)
    t = list(type_set)[0]
    return t, True


DEFAULT_ROOT = 'data'

def main():
    st.title('WSI Analysis System')

    if 'current_dir' not in st.session_state:
        st.session_state.current_dir = DEFAULT_ROOT

    if st.button('↑ Parent Directory'):
        st.session_state.current_dir = parent_dir
        st.rerun()

    files = list_files(st.session_state.current_dir)
    files_df = pd.DataFrame(files)

    st.subheader(f'File Selection')

    if len(files_df) ==  0:
        st.warning('This directory is empty.')
        return

    edited_df = st.data_editor(
        files_df,
        column_config={
            'selected': st.column_config.CheckboxColumn(
                label='✓',
                width='small',
                # help='Select files'
            ),
            'name': 'File Name',
            'type': 'Type',
            'size_mb': 'Size',
            'modified': 'Last Modified',
            'path': None,  # Hide path column
        },
        hide_index=True,
        use_container_width=True,
        # selection_mode='single-row',
        disabled=['name', 'type', 'size_mb', 'modified'],
        # selection_mode='multi-row',
    )

    selected_files = edited_df[edited_df['selected'] == True].to_dict('records')

    mode, multi = get_mode_and_multi(selected_files)
    if mode == 'Empty':
        st.write('Select files from checkboxes.')
    elif mode == 'Directory':
        if multi:
            st.warning('Multi directories are seleted.')
        else:
            if st.button('Move to this directory'):
                st.session_state.current_dir = selected_files[0].path
                st.rerun()

    elif mode == 'Other':
        st.warning('Some selected files are not supported. Only WSI (.ndpi, .svs) and HDF5 (.h5) files can be processed.')
    elif mode == 'Mix':
        st.warning('Cannot mix WSI and HDF5 files in the same operation. Please select only one type.')
    elif mode == 'Wsi':
        st.subheader('Processing Options')
        st.write('WSI files selected. Available operations:')

        # Operation options
        operation = st.radio(
            'Choose operation',
            ['Convert to h5 + Extract features', 'Convert to h5 + Extract features + Clustering'],
            index=1
        )

        # For multiple files, need a cluster name
        cluster_name = None
        if len(wsi_files) > 1 or 'Clustering' in operation:
            cluster_name = st.text_input('Cluster name for multiple files (required)', key='wsi_cluster_name')

        if st.button('Execute Processing', key='process_wsi'):
            if len(wsi_files) > 1 and (not cluster_name or cluster_name.strip() == '') and 'Clustering' in operation:
                st.error('Cluster name is required for multiple files or when performing clustering')
            else:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process each file
                h5_paths = []
                for i, file in enumerate(wsi_files):
                    # Update progress
                    progress = int((i / len(wsi_files)) * 100)
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
        if len(h5_files) > 0:
            cluster_name = st.text_input('Cluster name (required for multiple files)', key='h5_cluster_name')

        if st.button('Perform Clustering', key='process_h5'):
            if len(h5_files) > 1 and (not cluster_name or cluster_name.strip() == ''):
                st.error('Cluster name is required for multiple files')
            else:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process each file
                h5_paths = [file['path'] for file in h5_files]

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
