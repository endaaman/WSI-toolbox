## WSIツールボックス

### 基本事項

WSIデータを様々な形で活用する
- 形式を問わず、パッチ分割しhdf5に固める
- 基盤モデルなどに通して、パッチの埋め込みを取得
- クラスタリングおよびクラスタ番号を指定したサブクラスタリングなど包括的な解析を提供

## 開発に関して

- つねに　`uv` を使い、直接 `python` `pip` を使わない
- `cli` は下記のように `pydantic-autocli` を使ってサブコマンドベースのコマンドラインツールとしている
- **importは必ずファイル先頭に書く**。関数内でのimportは禁止
- 未使用の変数・importは削除する

### Lint


```bash
uv run ruff check wsi_toolbox/ --fix
uv run ruff format wsi_toolbox/
```

### リリース / バージョニング

- バージョンは `pyproject.toml` の `version` で管理（`wsi_toolbox/__init__.py` は `importlib.metadata` から取得）
- バージョンを上げるコミットは `Bump version X.Y.Z` の形式にならう
- **git tag は不要**（打たない）
- PyPI への公開は `./deploy.sh` を使う（clean → build → `twine check` → `y/N` 確認 → upload）
  - `~/.pypirc` の `[pypi]` トークンで認証
  - **PyPI への upload は Claude が勝手に実行しない。ken 本人が実行する**（`! ./deploy.sh`）
- slide-level encoding (TITAN 等) は uv-only 機能で PyPI パッケージには含めない（`pyproject.toml` 参照）

### AutoCLI の使い方

Key patterns:
- `def run_foo_bar(self, args):` → `python script.py foo-bar`
- `def prepare(self, args):` → shared initialization  
- `class FooBarArgs(AutoCLI.CommonArgs):` → command arguments
- Return `True`/`None` (success), `False` (fail), `int` (exit code)

For details: `python your_script.py --help`
