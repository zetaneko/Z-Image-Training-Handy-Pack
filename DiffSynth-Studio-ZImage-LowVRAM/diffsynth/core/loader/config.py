import torch, glob, os
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_source: str = None
    local_model_path: str = None
    skip_download: bool = None
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    onload_device: Optional[Union[str, torch.device]] = None
    onload_dtype: Optional[torch.dtype] = None
    preparing_device: Optional[Union[str, torch.device]] = None
    preparing_dtype: Optional[torch.dtype] = None
    computation_device: Optional[Union[str, torch.device]] = None
    computation_dtype: Optional[torch.dtype] = None
    clear_parameters: bool = False

    @staticmethod
    def _is_huggingface_cache_path(path: str) -> bool:
        """Check if path is a HuggingFace cache directory."""
        path_obj = Path(path)
        # HF cache structure: .../hub/models--org--model/snapshots/hash/
        return 'models--' in str(path_obj) and 'snapshots' in str(path_obj).split(os.sep)

    @staticmethod
    def _resolve_huggingface_cache_path(base_path: str, model_id: str) -> Optional[str]:
        """
        Resolve HuggingFace cache path for a model ID.

        HF cache structure: {base_path}/models--{org}--{model}/snapshots/{hash}/
        This finds the latest snapshot automatically.
        """
        # Convert model_id (e.g., "Tongyi-MAI/Z-Image") to HF cache format
        hf_model_dir = model_id.replace('/', '--')
        hf_cache_dir = Path(base_path) / f'models--{hf_model_dir}'

        if not hf_cache_dir.exists():
            return None

        # Look for snapshots directory
        snapshots_dir = hf_cache_dir / 'snapshots'
        if not snapshots_dir.exists():
            return None

        # Find the most recent snapshot (by modification time)
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return None

        # Use the most recently modified snapshot
        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        return str(latest_snapshot)

    def check_input(self):
        if self.path is None and self.model_id is None:
            raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`. `skip_download=True` only supports the first one.""")
    
    def parse_original_file_pattern(self):
        if self.origin_file_pattern is None or self.origin_file_pattern == "":
            return "*"
        elif self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        else:
            return self.origin_file_pattern
        
    def parse_download_source(self):
        if self.download_source is None:
            if os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE') is not None:
                return os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE')
            else:
                return "modelscope"
        else:
            return self.download_source
        
    def parse_skip_download(self):
        if self.skip_download is None:
            if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD') is not None:
                if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD').lower() == "true":
                    return True
                elif os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD').lower() == "false":
                    return False
            else:
                return False
        else:
            return self.skip_download

    def download(self):
        origin_file_pattern = self.parse_original_file_pattern()
        downloaded_files = glob.glob(origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
        download_source = self.parse_download_source()
        if download_source.lower() == "modelscope":
            snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False
            )
        elif download_source.lower() == "huggingface":
            hf_snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")
        
    def require_downloading(self):
        if self.path is not None:
            return False
        skip_download = self.parse_skip_download()
        return not skip_download
    
    def reset_local_model_path(self):
        if os.environ.get('DIFFSYNTH_MODEL_BASE_PATH') is not None:
            self.local_model_path = os.environ.get('DIFFSYNTH_MODEL_BASE_PATH')
        elif self.local_model_path is None:
            self.local_model_path = "./models"

        # Expand user path (e.g., ~/.cache -> /home/user/.cache)
        self.local_model_path = os.path.expanduser(self.local_model_path)

    def download_if_necessary(self):
        self.check_input()
        self.reset_local_model_path()

        # If path is not set, try to resolve it
        if self.path is None and self.model_id is not None:
            # First, try to find in HuggingFace cache format
            hf_cache_path = self._resolve_huggingface_cache_path(self.local_model_path, self.model_id)

            if hf_cache_path:
                # Found in HF cache, use that path
                if self.origin_file_pattern:
                    pattern = self.parse_original_file_pattern()

                    # Check if pattern is a directory (ends with /)
                    if self.origin_file_pattern.endswith('/'):
                        # For directory patterns (like "tokenizer/"), use the directory path itself
                        dir_path = os.path.join(hf_cache_path, self.origin_file_pattern.rstrip('/'))
                        if os.path.isdir(dir_path):
                            self.path = dir_path
                            print(f"Using HuggingFace cache: {hf_cache_path}")
                        else:
                            # Directory doesn't exist in cache, fall through to download
                            pass
                    else:
                        # For file patterns (like "*.safetensors"), glob for files
                        matched_files = glob.glob(os.path.join(hf_cache_path, pattern))
                        if matched_files:
                            self.path = matched_files
                            print(f"Using HuggingFace cache: {hf_cache_path}")
                        else:
                            # Pattern didn't match in HF cache, fall through to download
                            pass
                else:
                    self.path = hf_cache_path
                    print(f"Using HuggingFace cache: {hf_cache_path}")

        # If still no path, proceed with download or standard path resolution
        if self.path is None:
            if self.require_downloading():
                self.download()

            # Try standard path format
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                self.path = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))

        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]

    def vram_config(self):
        return {
            "offload_device": self.offload_device,
            "offload_dtype": self.offload_dtype,
            "onload_device": self.onload_device,
            "onload_dtype": self.onload_dtype,
            "preparing_device": self.preparing_device,
            "preparing_dtype": self.preparing_dtype,
            "computation_device": self.computation_device,
            "computation_dtype": self.computation_dtype,
        }
