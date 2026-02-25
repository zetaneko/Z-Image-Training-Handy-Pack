"""
Google Drive sync for zitpack dataset files.

Downloads missing .zitpack files from Google Drive to a local directory before
training, so remote workers can start with a single command — no manual file
transfers needed.

Two supported methods:

  rclone (recommended for personal Google accounts):
    - Install rclone (https://rclone.org/install/)
    - Run 'rclone config' to add a Google Drive remote (one-time, interactive)
    - Copy ~/.config/rclone/rclone.conf to remote workers — no re-auth needed
    - rclone handles incremental sync natively (skips existing files)

  Service account (for GCP/automated setups):
    - Create a service account in Google Cloud Console
    - Download the JSON key file
    - Share the Drive folder with the service account's email address
    - Pass the JSON path via --gdrive_credentials
    - Only downloads files not already present locally

Usage:
    from gdrive_sync import sync_zitpacks

    # Via rclone (personal account)
    sync_zitpacks(local_dir="/data/zitpacks", rclone_remote="gdrive:MyDatasets/zitpacks")

    # Via service account (private folder)
    sync_zitpacks(
        local_dir="/data/zitpacks",
        gdrive_folder_id="1AbCdEfGhIjKlMnOpQrStUvWxYz",
        gdrive_credentials="/path/to/service-account.json",
    )
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def sync_via_rclone(remote_path: str, local_dir: Path) -> None:
    """
    Sync .zitpack files from an rclone remote to a local directory.

    Uses 'rclone copy', which skips files that already exist locally
    (matched by size and modification time). Only .zitpack files are transferred.

    Args:
        remote_path: rclone source path, e.g. "gdrive:MyFolder/zitpacks".
        local_dir:   Local destination directory (created if it doesn't exist).
    """
    if not shutil.which("rclone"):
        raise RuntimeError(
            "rclone is not installed or not in PATH.\n"
            "Install from https://rclone.org/install/ then configure with: rclone config"
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Syncing .zitpack files: {remote_path} → {local_dir}")
    subprocess.run(
        [
            "rclone", "copy",
            remote_path, str(local_dir),
            "--include", "*.zitpack",
            "--progress",
        ],
        check=True,
    )
    zitpacks = sorted(local_dir.glob("*.zitpack"))
    print(f"Sync complete — {len(zitpacks)} .zitpack file(s) in {local_dir}.")


def sync_via_service_account(
    folder_id: str,
    local_dir: Path,
    credentials_path: str,
) -> None:
    """
    Download missing .zitpack files from a Google Drive folder using a service account.

    The Drive folder must be shared with the service account's email address
    (the 'client_email' field in the credentials JSON).

    Requires: pip install google-api-python-client google-auth

    Args:
        folder_id:        Google Drive folder ID (from the URL).
        local_dir:        Local destination directory (created if it doesn't exist).
        credentials_path: Path to the service account credentials JSON file.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        raise ImportError(
            "Google API packages are required for service account access.\n"
            "Install with: pip install google-api-python-client google-auth"
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in local_dir.glob("*.zitpack")}

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    # List all .zitpack files in the folder (paginated)
    print(f"Listing .zitpack files in Google Drive folder {folder_id} ...")
    all_files = []
    page_token = None
    while True:
        kwargs = dict(
            q=f"'{folder_id}' in parents and name contains '.zitpack' and trashed=false",
            fields="nextPageToken, files(id, name, size)",
            pageSize=1000,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.files().list(**kwargs).execute()
        all_files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    zitpack_files = [f for f in all_files if f["name"].endswith(".zitpack")]
    print(f"Found {len(zitpack_files)} .zitpack file(s) in Drive.")

    to_download = [f for f in zitpack_files if f["name"] not in existing]
    already_have = len(zitpack_files) - len(to_download)

    if already_have:
        print(f"  {already_have} file(s) already exist locally — skipping.")
    if not to_download:
        print("Nothing to download.")
        return

    for file_meta in to_download:
        name = file_meta["name"]
        size_bytes = int(file_meta.get("size", 0))
        size_mb = size_bytes / 1024 / 1024
        dest = local_dir / name

        print(f"  Downloading {name} ({size_mb:.1f} MB) ...")
        request = service.files().get_media(fileId=file_meta["id"])
        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=32 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"    {int(status.progress() * 100):3d}%", end="\r")
        print(f"    {name} — done.")

    print(f"Sync complete — downloaded {len(to_download)} file(s).")


def _strip_shell_quotes(s: Optional[str]) -> Optional[str]:
    """Remove a single layer of surrounding shell quotes if present.

    Shell scripts that build arguments with $(echo "--flag \"$VAR\"") pass literal
    quote characters to Python (bash does not strip quotes from command-substitution
    output). This helper undoes that so callers receive the bare value.
    """
    if s is None:
        return None
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    return s


def download_file_from_remote(remote_path: str, local_dir: Path) -> Path:
    """
    Download a single file from an rclone remote to a local directory.

    Skips the download if the file already exists locally.

    Args:
        remote_path: Full rclone path to the file, e.g.
                     "gdrive:checkpoints/model_step_5000.safetensors".
        local_dir:   Local directory to download the file into (created if absent).

    Returns:
        Path to the local copy of the file.
    """
    if not shutil.which("rclone"):
        raise RuntimeError(
            "rclone is not installed or not in PATH.\n"
            "Install from https://rclone.org/install/ then configure with: rclone config"
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    filename = remote_path.rsplit("/", 1)[-1]
    local_file = local_dir / filename

    if local_file.exists():
        print(f"Checkpoint already cached locally: {local_file}")
        return local_file

    print(f"Downloading checkpoint: {remote_path} → {local_file}")
    subprocess.run(
        ["rclone", "copy", remote_path, str(local_dir), "--progress"],
        check=True,
    )

    if not local_file.exists():
        raise FileNotFoundError(
            f"rclone reported success but file not found at: {local_file}\n"
            f"Check that the remote path is correct: {remote_path}"
        )

    print(f"Downloaded: {local_file}")
    return local_file


def sync_zitpacks(
    local_dir: str,
    rclone_remote: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
    gdrive_credentials: Optional[str] = None,
) -> None:
    """
    Sync .zitpack files from Google Drive to a local directory before training.

    Only files not already present locally will be downloaded.

    Priority: rclone_remote > gdrive_folder_id + gdrive_credentials.
    If neither is set, this is a no-op (no sync needed).

    Args:
        local_dir:          Local directory to sync files into.
        rclone_remote:      rclone source path, e.g. "gdrive:MyFolder/zitpacks".
                            Best for personal Google accounts — configure once with
                            'rclone config', then copy rclone.conf to remote workers.
        gdrive_folder_id:   Google Drive folder ID. Requires gdrive_credentials.
        gdrive_credentials: Path to a service account JSON key file.
                            The Drive folder must be shared with the account's email.
    """
    # Strip surrounding shell quotes that leak through $(echo "--flag \"$VAR\"") patterns
    local_dir = _strip_shell_quotes(local_dir)
    rclone_remote = _strip_shell_quotes(rclone_remote)
    gdrive_folder_id = _strip_shell_quotes(gdrive_folder_id)
    gdrive_credentials = _strip_shell_quotes(gdrive_credentials)

    if not rclone_remote and not gdrive_folder_id:
        return  # No sync configured

    local_path = Path(local_dir)

    if rclone_remote:
        sync_via_rclone(rclone_remote, local_path)
    elif gdrive_folder_id and gdrive_credentials:
        sync_via_service_account(gdrive_folder_id, local_path, gdrive_credentials)
    elif gdrive_folder_id:
        raise ValueError(
            "--gdrive_credentials (service account JSON) is required when using --gdrive_folder_id.\n"
            "For personal Google accounts, use --rclone_remote instead:\n"
            "  1. Install rclone: https://rclone.org/install/\n"
            "  2. Configure: rclone config\n"
            "  3. Set RCLONE_REMOTE=\"gdrive:YourFolder/zitpacks\" in the shell script"
        )
