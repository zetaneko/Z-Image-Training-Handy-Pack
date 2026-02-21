import torch, os, sys, argparse, accelerate, copy
from pathlib import Path
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ZImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        self.pipe = ZImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        if task == "trajectory_imitation":
            # This is an experimental feature.
            # We may remove it in the future.
            self.loss_fn = TrajectoryImitationLoss()
            self.task_to_loss["trajectory_imitation"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        if self.task == "trajectory_imitation":
            inputs_shared["cfg_scale"] = 2
            inputs_shared["teacher"] = self.pipe_teacher
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def z_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    return parser


if __name__ == "__main__":
    parser = z_image_parser()
    args = parser.parse_args()

    # Strip literal shell quotes that leak through $(echo "--flag \"$VAR\"") patterns
    def _sq(s):
        if s is None:
            return None
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            s = s[1:-1]
        return s

    args.zitpacks = _sq(args.zitpacks)
    args.zitpack_repeats = _sq(args.zitpack_repeats)
    args.dataset_base_path = _sq(args.dataset_base_path)
    args.dataset_metadata_path = _sq(args.dataset_metadata_path)
    args.model_base_path = _sq(args.model_base_path)
    args.rclone_remote = _sq(args.rclone_remote)
    args.gdrive_folder_id = _sq(args.gdrive_folder_id)
    args.gdrive_credentials = _sq(args.gdrive_credentials)

    # Set custom model base path if provided
    if args.model_base_path is not None:
        os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = args.model_base_path
        print(f"Using custom model base path: {args.model_base_path}")

    # Validate dataset source
    if not args.zitpacks and not args.dataset_base_path:
        print("Error: Either --zitpacks or --dataset_base_path must be provided.")
        sys.exit(1)

    # Add python-scripts to path for local utilities
    scripts_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "python-scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Sync zitpack files from Google Drive — rank 0 only to avoid concurrent rclone corruption
    if args.zitpacks and (args.rclone_remote or args.gdrive_folder_id):
        if accelerator.is_main_process:
            from gdrive_sync import sync_zitpacks
            sync_zitpacks(
                local_dir=args.zitpacks,
                rclone_remote=args.rclone_remote,
                gdrive_folder_id=args.gdrive_folder_id,
                gdrive_credentials=args.gdrive_credentials,
            )
        accelerator.wait_for_everyone()  # all ranks wait for rank 0 to finish downloading

    if args.zitpacks:
        # Load dataset from .zitpack archives
        zitpack_dir = Path(args.zitpacks)
        zitpack_files = sorted(zitpack_dir.glob("*.zitpack"))
        if not zitpack_files:
            print(f"Error: No .zitpack files found in {zitpack_dir}")
            sys.exit(1)

        from dataset_archive import ZitpackDataset, parse_zitpack_repeats

        _resizer = ImageCropAndResize(
            max_pixels=args.max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )
        archive_repeats = parse_zitpack_repeats(zitpack_files, args.zitpack_repeats)
        _base_dataset = ZitpackDataset(zitpack_files, repeat=args.dataset_repeat, archive_repeats=archive_repeats)

        class _ResizedZitpackDataset:
            load_from_cache = False
            def __init__(self, ds, resizer):
                self._ds = ds
                self._resizer = resizer
            def __len__(self):
                return len(self._ds)
            def __getitem__(self, i):
                d = self._ds[i]
                d["image"] = self._resizer(d["image"])
                return d

        dataset = _ResizedZitpackDataset(_base_dataset, _resizer)
        print(f"Loaded {_base_dataset.archive_count} zitpack archive(s) with {_base_dataset.total_entries} unique entries ({_base_dataset.weighted_entries} weighted)")
        for f, r in zip(zitpack_files, archive_repeats):
            print(f"  - {f.name}  (repeat x{r})")
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            )
        )
    model = ZImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "trajectory_imitation": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
