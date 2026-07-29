from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional helper: download LUNA16 subset0 from a public HF mirror."
    )
    parser.add_argument("--local-dir", default="data/LUNA16")
    parser.add_argument("--repo-id", default="MedOtter/LUNA16")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "This optional downloader needs huggingface_hub. Install only this helper "
            "dependency with: pip install huggingface_hub"
        ) from error

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    items = list(
        api.list_repo_tree(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo="subset0",
            recursive=True,
            expand=False,
        )
    )
    filenames = sorted(
        item.path for item in items if getattr(item, "path", "").startswith("subset0/")
    )
    if not filenames:
        raise RuntimeError("The mirror returned no files under subset0/")
    print(f"Found {len(filenames)} files. Downloading to {local_dir}")

    def download(filename: str) -> str:
        return hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            endpoint=args.endpoint,
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, name): name for name in filenames}
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            completed += 1
            print(f"[{completed}/{len(filenames)}] {name}", flush=True)
    print(f"Download complete: {completed} files")


if __name__ == "__main__":
    main()
