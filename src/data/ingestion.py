"""
Data ingestion — downloads the dataset from Kaggle if it is not already
present locally.

Setup (one-time):
    1. Create a Kaggle account at kaggle.com
    2. Go to Account → API → Create New Token → saves kaggle.json
    3. Place kaggle.json at:
          Windows : C:\\Users\\<you>\\.kaggle\\kaggle.json
          Mac/Linux: ~/.kaggle/kaggle.json
       OR export the two environment variables:
          KAGGLE_USERNAME=your_username
          KAGGLE_KEY=your_api_key

Finding the dataset slug:
    Open the Kaggle dataset page. The URL looks like:
        kaggle.com/datasets/<owner>/<dataset-name>
    The slug is "<owner>/<dataset-name>".
    Set it in configs/config.yaml → kaggle.dataset_slug.
"""
import os
import shutil
import zipfile
from pathlib import Path


def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except ImportError:
        raise ImportError(
            'kaggle package not installed. Run: pip install kaggle'
        )
    except Exception as exc:
        raise RuntimeError(
            f'Kaggle authentication failed: {exc}\n\n'
            'Make sure kaggle.json is in ~/.kaggle/ or set the '
            'KAGGLE_USERNAME and KAGGLE_KEY environment variables.'
        ) from exc


def download_dataset(
    dataset_slug: str,
    destination_dir: str,
    filename: str,
    force: bool = False,
) -> Path:
    """
    Download `filename` from a Kaggle dataset if not already present.

    Parameters
    ----------
    dataset_slug  : Kaggle identifier in the form "owner/dataset-name".
    destination_dir: Local folder to place the CSV (e.g. "data/raw").
    filename      : Expected filename after extraction (e.g. "global_supply_chain_risk_2026.csv").
    force         : Re-download even if the file already exists.

    Returns
    -------
    Path to the local file.
    """
    dest_dir = Path(destination_dir)
    dest_file = dest_dir / filename

    if dest_file.exists() and not force:
        print(f'Dataset already present at {dest_file} — skipping download.')
        return dest_file

    print(f'Downloading dataset "{dataset_slug}" from Kaggle...')
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = _kaggle_api()

    # Download the full dataset as a zip into a temp subfolder
    tmp_dir = dest_dir / '_kaggle_tmp'
    tmp_dir.mkdir(exist_ok=True)
    try:
        api.dataset_download_files(
            dataset_slug,
            path=str(tmp_dir),
            unzip=False,
            quiet=False,
        )

        # Find and extract the zip
        zips = list(tmp_dir.glob('*.zip'))
        if not zips:
            raise FileNotFoundError(
                f'No zip file found after downloading "{dataset_slug}". '
                'The dataset may require a different download method.'
            )
        with zipfile.ZipFile(zips[0], 'r') as zf:
            # Try to extract just the target file if present, else extract all
            names = zf.namelist()
            if filename in names:
                zf.extract(filename, path=str(dest_dir))
            else:
                # Extract everything and look for the file
                zf.extractall(path=str(dest_dir))
                matches = list(dest_dir.rglob(filename))
                if matches and matches[0] != dest_file:
                    shutil.move(str(matches[0]), str(dest_file))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not dest_file.exists():
        raise FileNotFoundError(
            f'Expected file "{filename}" not found after extracting '
            f'"{dataset_slug}". Check that dataset_slug and filename '
            'in configs/config.yaml match the actual Kaggle dataset.'
        )

    print(f'Dataset saved → {dest_file}')
    return dest_file


def ensure_data_exists(config: dict, force: bool = False) -> Path:
    """
    Convenience wrapper: download the dataset if the local file is missing.
    Uses the kaggle block from config.yaml.
    """
    raw_path = Path(config['data']['raw_path'])
    if raw_path.exists() and not force:
        print(f'Data found at {raw_path}')
        return raw_path

    kaggle_cfg = config.get('kaggle', {})
    slug = kaggle_cfg.get('dataset_slug', '')
    if not slug or slug.startswith('<'):
        raise ValueError(
            'Dataset not found locally and no Kaggle slug configured.\n'
            'Set kaggle.dataset_slug in configs/config.yaml.\n'
            'Example:\n'
            '  kaggle:\n'
            '    dataset_slug: owner/dataset-name\n'
            '    filename: global_supply_chain_risk_2026.csv'
        )

    return download_dataset(
        dataset_slug=slug,
        destination_dir=str(raw_path.parent),
        filename=kaggle_cfg.get('filename', raw_path.name),
        force=force,
    )
