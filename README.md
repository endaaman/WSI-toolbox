# WSI Toolbox


## Python environment management

パッケージなどインストール

```
$ uv sync

# for flash-attn
$ uv sync --extra build
$ uv sync --extra build --extra compile
```



仮想環境に入る

```
$ sourve .venv/bin/activate
```



## Usage


### Convert WSI into hdf5

WSIファイルをパッチ分割してHDF5に固める

```
$ python -m wsi_toolbox.main wsi2h5 -i data/DLBCL-Morph/13952_0.svs -o out/13952_0.h5 --patch-size 256
```

指定したパッチサイズになるように0.4〜0.5mppでパッチ分割したものを座標とともに保存。メタデータにもいろいろ保存しておく。

```
'patches'                 : パッチのデータ dim:[<patch_count>, <patch_size[0]>, <patch_size[1]>, 3]
                            ex) [3237, 512, 512, 3] のようなテンソル
'coordinates'             : 各パッチのピクセル単位の座標 dim:[<patch_count>, 2]

'features'                : GigaPathで抽出した特等量 dim:[<patch_count>, 1536]

'clusters'                : DBSCANで取得したクラスタ番号 dim:[<patch_count>]

'metadata/original_mpp'   : もともとのmpp
'metadata/original_width' : もともとの画像の幅（level=0）
'metadata/original_height': もともとの画像の幅（level=0）
'metadata/image_level'    : 使ったレベル（基本的にはlevel=0になる）
'metadata/mpp'            : 出力されたパッチのmpp
'metadata/scale'          : 出力時のscale
'metadata/patch_size'     : パッチの解像度
'metadata/patch_count'    : パッチの総数
'metadata/cols'           : パッチを並べたときの横方向の数
'metadata/rows'           : パッチを並べたときの縦方向の数
```


### Calculate embeddings for each patch

GigaPathで各パッチの特徴量を抽出

```
$ python -m wsi_toolbox.main process-patches -i out/13952_0.h5 -B 256
```

デフォルトではGigaPathを使って`h5`ファイルの`gigapath/features` に書き込まれる。UNIを使うときは`--model uni`で`uni/features`


例）WSIをまとめてHDFに変換

```

$ ls data/DLBCL-Morph/*.svs | xargs -n1 basename | awk -F_ '{print $1" "$0}' | xargs -I{} sh -c 'set -- {}; pueue add python -m wsi_toolbox.main wsi2h5 --in "data/DLBCL-Morph/$2" --out "out/dataset/$1/${2%.svs}_256.h5" --patch-size 256'
```

`data/DLBCL-Morph/13952_0.svs` のように保存されているのを `out/dataset/13952/13952_0.h5` というような形式で保存する。

### Clustering

UMAPで次元削減し、HDBSCANでクラスタリング

```
$ python -m wsi_toolbox.main cluster -i out/13952_0.h5
```

デフォルトでは`gigapath/features`を読み取って、`gigapath/clusters` にクラスタ番号を保存


### Preview

白色でスキップされたパッチや、パッチのクラスタを確認

```
$ python -m wsi_toolbox.main preview -i out/13952_0.h5
```

`gigapath/clusters` に前段で計算したクラスタがあればその番号と色を反映したフレームも一緒に描画する。
