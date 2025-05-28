import os
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Set, Callable, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from .processor import WSIProcessor, TileProcessor, ClusterProcessor, PreviewClustersProcessor

class Status:
    REQUEST = "REQUEST"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    ERROR = "ERROR"
    
    @classmethod
    def is_processing_state(cls, status: str) -> bool:
        """状態が処理中系かどうかを判定"""
        return status.startswith((cls.PROCESSING, cls.DONE, cls.ERROR))

class Task:
    REQUEST_FILE = "_ROBIEMON.txt"
    LOG_FILE = "_ROBIEMON_LOG.txt"
    
    def __init__(self, folder: Path, on_complete: Optional[Callable[[Path], None]] = None):
        self.folder = folder
        self.status = Status.REQUEST
        self.on_complete = on_complete
        self.wsi_files = list(folder.glob("**/*.ndpi")) + list(folder.glob("**/*.svs"))
        self.wsi_files.sort()
    
    def write_banner(self):
        """処理開始時のバナーをログに書き込み"""
        self.append_log("\n" + "="*50 + "\n")
        self.append_log(f"Processing folder: {self.folder}\n")
        self.append_log(f"Found {len(self.wsi_files)} WSI files:\n")
        for i, wsi_file in enumerate(self.wsi_files, 1):
            size_mb = wsi_file.stat().st_size / (1024 * 1024)
            self.append_log(f"  {i}. {wsi_file.name} ({size_mb:.1f} MB)\n")
        self.append_log("="*50 + "\n")
    
    def run(self):
        try:
            # ログファイルをクリア
            with open(self.folder / self.LOG_FILE, "w") as f:
                f.write("")
            
            self.set_status(Status.PROCESSING)
            self.write_banner()
            
            # WSIファイルごとの処理
            for wsi_file in self.wsi_files:
                try:
                    self.append_log(f"\n{'='*30}\n")
                    self.append_log(f"Processing: {wsi_file.name}\n")
                    
                    # HDF5変換（既存の場合はスキップ）
                    h5_file = wsi_file.with_suffix(".h5")
                    if not h5_file.exists():
                        self.append_log("Converting to HDF5...\n")
                        wp = WSIProcessor(str(wsi_file))
                        wp.convert_to_hdf5(str(h5_file))
                        self.append_log("HDF5 conversion completed\n")
                    
                    # ViT特徴量抽出（既存の場合はスキップ）
                    self.append_log("Extracting ViT features...\n")
                    tp = TileProcessor(device="cuda")
                    tp.evaluate_hdf5_file(str(h5_file))
                    self.append_log("ViT feature extraction completed\n")
                    
                    # クラスタリングとUMAP生成
                    self.append_log("Starting clustering and UMAP generation...\n")
                    cp = ClusterProcessor([h5_file])
                    cp.anlyze_clusters(resolution=1.0)
                    cp.get_umap_embeddings()  # UMAP生成
                    
                    # UMAPプロット生成
                    base = str(wsi_file.with_suffix(""))
                    umap_path = f"{base}_umap.png"
                    cp.plot_umap(fig_path=umap_path)
                    self.append_log(f"UMAP plot saved to {os.path.basename(umap_path)}\n")
                    
                    # サムネイル生成
                    thumb_path = f"{base}_thumb.jpg"
                    thumb_proc = PreviewClustersProcessor(str(h5_file), size=64)
                    img = thumb_proc.create_thumbnail(cluster_name='')
                    img.save(thumb_path)
                    self.append_log(f"Thumbnail saved to {os.path.basename(thumb_path)}\n")
                    
                    self.append_log("Clustering and visualization completed\n")
                    
                except Exception as e:
                    self.append_log(f"Error processing {wsi_file}: {str(e)}\n")
                    self.set_status(Status.ERROR)
                    if self.on_complete:
                        self.on_complete(self.folder)
                    return
            
            self.set_status(Status.DONE)
            self.append_log("\nAll processing completed successfully\n")
            
        except Exception as e:
            self.append_log(f"Error: {str(e)}\n")
            self.set_status(Status.ERROR)
        
        if self.on_complete:
            self.on_complete(self.folder)
    
    def set_status(self, status: str):
        self.status = status
        with open(self.folder / self.REQUEST_FILE, "w") as f:
            f.write(f"{status}\n")
    
    def append_log(self, message: str):
        with open(self.folder / self.LOG_FILE, "a") as f:
            f.write(message)

class Watcher:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.running_tasks: Dict[Path, Task] = {}
        self.console = Console()
    
    def run(self, interval: int = 60):
        self.console.print("\n[bold blue]ROBIEMON Watcher started[/]")
        self.console.print(f"[blue]Watching directory:[/] {self.base_dir}")
        self.console.print(f"[blue]Polling interval:[/] {interval} seconds")
        self.console.print("[yellow]Press Ctrl+C to stop[/]\n")
        
        while True:
            try:
                self.check_folders()
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.remaining]{task.remaining}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Next check in", total=interval)
                    for _ in range(interval):
                        time.sleep(1)
                        progress.advance(task)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Stopping watcher...[/]")
                break
            except Exception as e:
                self.console.print(f"[red]ERROR:[/] {str(e)}")
    
    def check_folders(self):
        for folder in self.base_dir.rglob("*"):
            if not folder.is_dir():
                continue
                
            request_file = folder / Task.REQUEST_FILE
            if not request_file.exists():
                continue
                
            if folder in self.running_tasks:
                continue
                
            try:
                with open(request_file, "r") as f:
                    status = f.read().strip()
            except:
                continue
            
            if Status.is_processing_state(status):
                continue
                
            task = Task(folder, on_complete=lambda f: self.running_tasks.pop(f, None))
            self.running_tasks[folder] = task
            task.run()  # 同期実行に変更

BASE_DIR = os.getenv('BASE_DIR', 'data')

def main():
    parser = argparse.ArgumentParser(description="ROBIEMON WSI Processor Watcher")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR,
        help="Base directory to watch for WSI processing requests"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory '{args.base_dir}' does not exist")
        return
    if not base_dir.is_dir():
        print(f"Error: '{args.base_dir}' is not a directory")
        return
    
    watcher = Watcher(args.base_dir)
    watcher.run(interval=args.interval)  # asyncio.runを削除

if __name__ == "__main__":
    main()