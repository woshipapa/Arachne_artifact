import datetime
import functools
import json
import math
import os
import shutil
import time

import diffusers
import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DistributedType, skip_first_batches
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    release_memory,
    set_seed,
    tqdm,
)
from diffusers.utils import WEIGHTS_NAME
from vast.datasets import DefaultCollator
from vast.train.samplers import ParallelBatchSampler
from vast.models import utils as gm_utils
from vast.utils.acceleration import initialize_sequence_parallel_group
from vast.models.nn import ModuleDict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import BatchSampler

from .. import utils
from ..configs import load_config
from ..optimizers import build_optimizer
from ..samplers import build_sampler
from ..schedulers import build_scheduler
from vast.datasets.transforms import build_transform
from ..utils import EMAModel


class Trainer:
    def __init__(
        self,
        project_dir,
        max_epochs=0,
        max_steps=0,
        gradient_accumulation_steps=1,
        sp_size=1,
        mixed_precision=None,
        loss_nan_total_limit=100,
        checkpoint_interval=1,
        checkpoint_total_limit=-1,
        checkpoint_keeps=None,
        checkpoint_save_optimizer=False,
        log_with=None,
        log_interval=100,
        with_ema=False,
        activation_checkpointing=False,
        activation_class_names=None,
        find_unused_parameters=False,
        broadcast_buffers=True,
        allow_tf32=True,
        seed=6666,
        max_grad_norm=None,
        grad_norm_type=None,
        resume: bool = False,
        eval_interval: int = -1,
    ):
        assert seed > 0
        set_seed(seed)
        if project_dir.endswith("/"):
            project_dir = project_dir[:-1]
        project_name = os.path.basename(project_dir)
        project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=os.path.join(project_dir, "logs"),
        )
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=project_config,
            dataloader_config=dataloader_config,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=find_unused_parameters,
                    broadcast_buffers=broadcast_buffers,
                )
            ],
        )
        self.accelerator.init_trackers(project_name)
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            log_name = "train_{}.log".format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.loss_nan_total_limit = loss_nan_total_limit
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_total_limit = checkpoint_total_limit
        self.checkpoint_keeps = checkpoint_keeps
        self.checkpoint_save_optimizer = checkpoint_save_optimizer
        self.log_interval = log_interval
        self.activation_checkpointing = activation_checkpointing
        self.activation_class_names = activation_class_names
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.eval_interval = eval_interval

        self.with_ema = with_ema
        self.resume_flag = resume
        self.ema_models = []

        self._train_dataloaders = []
        self._eval_dataloaders = []
        self._models = []
        self._optimizers = []
        self._schedulers = []

        if max_epochs > 0:
            assert max_steps == 0
            by_epoch = True
        else:
            assert max_epochs > 0
            by_epoch = False
        self._by_epoch = by_epoch
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._cur_step = 0
        self._skip_batches = 0

        self._start_tic = None
        self._epoch_tic = None
        self._step_tic = None
        self._data_time = 0
        self._outputs = dict()
        self._loss_nan_count = 0

        self.sp_size = sp_size
        if self.sp_size > 1:
            assert dist.get_world_size() == self.num_processes
            assert dist.get_rank() == self.process_index
            initialize_sequence_parallel_group(self.sp_size)

        if self.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "zero_force_ds_cpu_optimizer"
            ] = False
        self.accelerator.register_for_checkpointing(self)
        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    @property
    def project_dir(self):
        return self.accelerator.project_dir

    @property
    def logging_dir(self):
        return self.accelerator.logging_dir

    @property
    def model_dir(self):
        return os.path.join(self.project_dir, "models")

    @property
    def distributed_type(self):
        return self.accelerator.distributed_type

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def process_index(self):
        return self.accelerator.process_index

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    @property
    def is_last_process(self):
        return self.accelerator.is_last_process

    @property
    def mixed_precision(self):
        return self.accelerator.mixed_precision

    @property
    def device(self):
        return self.accelerator.device

    @property
    def dtype(self):
        if self.mixed_precision == "fp16":
            return torch.float16
        if self.mixed_precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def gradient_accumulation_steps(self):
        return self.accelerator.gradient_accumulation_steps

    @property
    def train_dataloaders(self):
        return self._train_dataloaders

    @property
    def train_dataloader(self):
        return self._train_dataloaders[0]

    @property
    def eval_dataloaders(self):
        return self._eval_dataloaders

    @property
    def eval_dataloader(self):
        if self._eval_dataloaders:
            return self._eval_dataloaders[0]
        return None

    @property
    def models(self):
        return self._models

    @property
    def model(self):
        return self._models[0]

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def optimizer(self):
        return self._optimizers[0]

    @property
    def schedulers(self):
        return self._schedulers

    @property
    def scheduler(self):
        return self._schedulers[0]

    @property
    def train_data_size(self):
        return len(self.train_dataloader.dataset)

    @property
    def eval_data_size(self):
        if self.eval_dataloaders:
            return len(self.eval_dataloader.dataset)
        return 0

    @property
    def batch_size(self):
        if self.train_dataloader.batch_sampler is not None:
            batch_sampler = self.train_dataloader.batch_sampler
        else:
            batch_sampler = self.train_dataloader.sampler
        while True:
            if hasattr(batch_sampler, "batch_sampler"):
                batch_sampler = batch_sampler.batch_sampler
            else:
                break
        if hasattr(batch_sampler, "batch_size"):
            batch_size = batch_sampler.batch_size
        elif hasattr(batch_sampler, "batch_sizes"):
            batch_size = min(batch_sampler.batch_sizes)
        else:
            assert False
        return (
            batch_size
            * self.num_processes
            * self.gradient_accumulation_steps
            // self.sp_size
        )

    @property
    def epoch_size(self):
        # print(f"dataloader._index_sampler={self.train_dataloader._index_sampler}")
        # return len(self.train_dataloader)

        return int(
            math.ceil(
                (len(self.train_dataloader) + self._skip_batches)
                / self.gradient_accumulation_steps
            )
        )

    @property
    def max_epochs(self):
        if self._max_epochs > 0:
            return self._max_epochs
        else:
            return int(math.ceil(self._max_steps / self.epoch_size))

    @property
    def max_steps(self):
        if self._max_steps > 0:
            return self._max_steps
        else:
            return self._max_epochs * self.epoch_size

    @property
    def cur_epoch(self):
        return int(math.ceil(self.cur_step / self.epoch_size))

    @property
    def cur_step(self):
        return self._cur_step

    def print(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    def state_dict(self):
        return {"step": self._cur_step}

    def load_state_dict(self, state_dict):
        self._cur_step = state_dict["step"]

    @classmethod
    def load(cls, config_or_path):
        config = load_config(config_or_path).copy()
        trainer = cls(project_dir=config.project_dir, **config.train)
        trainer.prepare(
            train_dataloaders=config.dataloaders.train,
            models=config.models.train
            if hasattr(config.models, "train")
            else config.models,
            optimizers=config.optimizers,
            schedulers=config.schedulers,
            eval_dataloaders=config.dataloaders.eval
            if hasattr(config.dataloaders, "eval")
            else None,
        )
        return trainer

    def save_config(self, config):
        if not self.is_main_process:
            return
        config = load_config(config)
        config_path = os.path.join(self.project_dir, "config.json")
        config.save(config_path)

    def load_checkpoint(self, checkpoint, models, strict=True):
        if checkpoint is None:
            return
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        if not isinstance(models, list):
            models = [models]
        for i in range(len(checkpoint)):
            config_path = os.path.join(checkpoint[i], "config.json")
            config = json.load(open(config_path, "r"))
            class_name = config["_class_name"]
            self.logger.info(f"Load {class_name} from {checkpoint[i]}")
            state_dict = gm_utils.load_state_dict(checkpoint[i])
            flag = False
            for model in models:
                if model.__class__.__name__ == class_name:
                    mes = model.load_state_dict(state_dict, strict=strict)
                    if self.is_main_process and not strict:
                        self.logger.info(mes)
                    flag = True
                    break
            if not flag:
                raise ValueError("No model loaded by {checkpoint[i]}")

    def get_checkpoint(self):
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
        checkpoints = [
            os.path.join(self.model_dir, checkpoint) for checkpoint in checkpoints
        ]
        return checkpoints

    def remove_checkpoint(self, total_limit=None):
        if not self.is_main_process:
            return
        total_limit = total_limit or self.checkpoint_total_limit
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))
        if self.checkpoint_keeps is not None:
            new_checkpoints = []
            for checkpoint in checkpoints:
                if self._by_epoch:
                    checkpoint_id = int(checkpoint.split("_")[-3])
                else:
                    checkpoint_id = int(checkpoint.split("_")[-1])
                if checkpoint_id not in self.checkpoint_keeps:
                    new_checkpoints.append(checkpoint)
            checkpoints = new_checkpoints
        if len(checkpoints) >= total_limit > 0:
            num_to_remove = len(checkpoints) - self.checkpoint_total_limit + 1
            for checkpoint in checkpoints[:num_to_remove]:
                checkpoint = os.path.join(self.model_dir, checkpoint)
                self.logger.info("Remove checkpoint {}".format(checkpoint))
                shutil.rmtree(checkpoint)
            checkpoints = checkpoints[num_to_remove:]
        if not self.checkpoint_save_optimizer:
            if len(checkpoints) > 0:
                checkpoints = checkpoints[:-1]
            for checkpoint in checkpoints:
                if self.distributed_type == DistributedType.DEEPSPEED:
                    checkpoint = os.path.join(
                        self.model_dir, checkpoint, "pytorch_model"
                    )
                    if os.path.exists(checkpoint):
                        shutil.rmtree(checkpoint)
                else:
                    for i in range(len(self.optimizers)):
                        optimizer_name = (
                            "optimizer.bin" if i == 0 else f"optimizer_{i}.bin"
                        )
                        checkpoint_i = os.path.join(
                            self.model_dir, checkpoint, optimizer_name
                        )
                        if os.path.exists(checkpoint_i):
                            os.remove(checkpoint_i)

    def resume(self):
        checkpoints = (
            self.get_checkpoint()
        )  # return list of checkpoint under the project_dir/models/
        if len(checkpoints) == 0:
            return
        succeed_resume_flag = False
        for checkpoint in checkpoints[::-1]:
            try:
                self.accelerator.load_state(checkpoint)
                print("Succeed load checkpoint at :", checkpoint)
                succeed_resume_flag = True
                break
            except Exception:
                print("Error in load checkpoint at: ", checkpoint)
        if not succeed_resume_flag:
            return

        if self.train_dataloader.batch_sampler is not None:
            sampler = self.train_dataloader.batch_sampler
        else:
            sampler = self.train_dataloader.sampler
        while True:
            if hasattr(sampler, "batch_sampler"):
                sampler = sampler.batch_sampler
            elif hasattr(sampler, "sampler"):
                sampler = sampler.sampler
            else:
                break
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(math.floor(self.cur_step / self.epoch_size)))
            skip_batches = (
                self.cur_step % self.epoch_size
            ) * self.gradient_accumulation_steps // self.num_processes
        else:
            skip_batches = self.cur_step * self.gradient_accumulation_steps // self.num_processes
        if skip_batches > 0:
            for i in range(len(self._train_dataloaders)):
                max_skips = len(self._train_dataloaders[i])
                actual_skips = min(skip_batches, max_skips)
                if actual_skips < skip_batches:
                    self.logger.warning(
                        f"skip_batches {skip_batches} exceeds the dataloader length {max_skips}; actually skipped {actual_skips} batches."
                    )
                self.logger.warning(f"before skip_first_batches len(self._train_dataloaders[{i}])={len(self._train_dataloaders[i])}")
                self._train_dataloaders[i] = skip_first_batches(
                    self._train_dataloaders[i], actual_skips
                )
                self.logger.warning(f"After skip_first_batches len(self._train_dataloaders[{i}])={len(self._train_dataloaders[i])}")

            skip_batches = actual_skips
        self._skip_batches = skip_batches

    def timeit_context(self, name):
        class TimeIt:
            def __init__(self, trainer, name):
                self.trainer = trainer
                self.name = name
                self.local_rank = trainer.local_process_index
                self.global_rank = trainer.process_index
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                self.trainer.logger.info(
                    f"[Local Rank {self.local_rank}] [Global Rank {self.global_rank}] {self.name} Elapsed Time: {elapsed_time:.2f} seconds"
                )

        return TimeIt(self, name)

    def get_dataloader(self, data_config):
        from vast.datasets.datasets.build import build_dataset

        batch_size_per_gpu = data_config.get("batch_size_per_gpu", 1)
        batch_size = (
            batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        )
        dataset = build_dataset(data_config.dataset)
        filter_cfg = data_config.get("filter", None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform_cfg = data_config.get("transform", None)
        if transform_cfg is not None:
            transform = build_transform(data_config.transform)
            dataset.set_transform(transform)
        if "batch_sampler" in data_config:
            batch_sampler_cfg = data_config.batch_sampler
            batch_sampler = build_sampler(
                batch_sampler_cfg,
                dataset=dataset,
                batch_size_per_gpu=batch_size_per_gpu,
                batch_size=batch_size,
            )
        else:
            sampler_cfg = data_config.get("sampler", {"type": "DefaultSampler"})
            sampler = build_sampler(sampler_cfg, dataset=dataset, batch_size=batch_size)
            batch_sampler = BatchSampler(
                sampler, batch_size=batch_size_per_gpu, drop_last=False
            )
        if self.sp_size > 1:
            batch_sampler = ParallelBatchSampler(batch_sampler, sp_size=self.sp_size)
        collator = data_config.get("collator", {})
        collator = DefaultCollator(**collator)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=data_config.num_workers,
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            if getattr(batch_sampler, "batch_size", None) is not None:
                batch_size = batch_sampler.batch_size
            elif getattr(batch_sampler, "batch_sizes", None) is not None:
                batch_size = min(batch_sampler.batch_sizes)
            else:
                assert False
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = batch_size
        if getattr(batch_sampler, "batch_size", None) is None:
            self.accelerator.even_batches = False
        return dataloader

    def get_models(self, *args, **kwargs):
        raise NotImplementedError

    def get_optimizers(self, optimizers):
        optimizers = utils.as_list(optimizers)
        for i in range(len(optimizers)):
            if isinstance(optimizers[i], dict):
                if len(optimizers) == 1 and len(self.models) > 1:
                    params = []
                    for model in self.models:
                        params += list(
                            [p for p in model.parameters() if p.requires_grad]
                        )
                elif len(optimizers) == len(self.models):
                    params = [p for p in self.models[i].parameters() if p.requires_grad]
                else:
                    assert False
                # import pdb; pdb.set_trace()
                optimizers[i] = build_optimizer(optimizers[i], params=params)
        return optimizers

    def get_schedulers(self, schedulers):
        schedulers = utils.as_list(schedulers)
        assert len(schedulers) == len(self.optimizers)
        for i in range(len(schedulers)):
            if isinstance(schedulers[i], dict):
                schedulers[i] = build_scheduler(
                    schedulers[i],
                    optimizer=self.optimizers[i],
                    epoch_size=self.epoch_size,
                    max_epochs=self.max_epochs,
                    max_steps=self.max_steps,
                )
        return schedulers

    def set_ema_models(self):
        if self.with_ema:
            for model in self.models:
                ema_model = EMAModel(
                    rank=self.process_index, world_size=self.num_processes
                )
                ema_model.load_state_dict(model.state_dict())
                self.ema_models.append(ema_model)

    def apply_activation_checkpointing(self, models=None):
        if self.activation_checkpointing and self.activation_class_names is not None:
            models = models or self.models
            for model in models:
                cls_to_wrap = set()
                for class_name in self.activation_class_names:
                    for module in model.modules():
                        if module.__class__.__name__ == class_name:
                            cls_to_wrap.add(module.__class__)
                            break
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=cls_to_wrap
                )
                apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)

    def prepare(
        self, train_dataloaders, models, optimizers, schedulers, eval_dataloaders=None
    ):
        with self.timeit_context("Get ALL dataloaders"):
            self._train_dataloaders = utils.as_list(
                self.get_dataloader(train_dataloaders)
            )
        with self.timeit_context("Get ALL models"):
            self._models = utils.as_list(self.get_models(models))
        self.set_ema_models()
        self.apply_activation_checkpointing()
        if self.distributed_type == DistributedType.FSDP:
            self._models = utils.as_list(self.accelerator.prepare(*self._models))
        self._optimizers = utils.as_list(self.get_optimizers(optimizers))
        self._schedulers = utils.as_list(self.get_schedulers(schedulers))
        if self.distributed_type == DistributedType.FSDP:
            objects = [self._train_dataloaders, self._optimizers, self._schedulers]
        else:
            objects = [
                self._train_dataloaders,
                self._models,
                self._optimizers,
                self._schedulers,
            ]
        if eval_dataloaders is not None:
            self._eval_dataloaders = utils.as_list(
                self.get_dataloader(eval_dataloaders)
            )
            objects.append(self._eval_dataloaders)
        inputs = functools.reduce(lambda x, y: x + y, objects)
        outputs = utils.as_list(self.accelerator.prepare(*inputs))
        start_idx = 0
        for obj in objects:
            end_idx = start_idx + len(obj)
            obj[:] = outputs[start_idx:end_idx]
            start_idx = end_idx

    def train(self):
        release_memory()
        if self.resume_flag:
            self.resume()
        self.print_before_train()
        dataloader_iter = iter(self.train_dataloader)
        while self._cur_step < self.max_steps:
            self._cur_step += 1
            self.model.train()
            for _ in range(self.gradient_accumulation_steps):
                data_start_tic = time.time()
                batch_dict = next(dataloader_iter)
                self._data_time = time.time() - data_start_tic
                with self.accelerator.accumulate(*self.models):
                    losses = self.forward_step(batch_dict)
                    loss = self.parse_losses(losses)
                    self.backward_step(loss)
            self.print_step()
            self.eval()
            self.save_checkpoint_step()
        self.print_after_train()
        self.accelerator.end_training()
        release_memory()

    def eval(self):
        if (
            self.eval_interval > 0
            and self.cur_step % self.eval_interval == 0
            and self.cur_step > 0
            or self.cur_step == self.max_steps
        ):
            if self.eval_dataloader is not None:
                self._outputs.clear()
                self.model.eval()
                for batch_dict in tqdm(self.eval_dataloader):
                    with torch.no_grad():
                        losses = self.forward_step(batch_dict)
                        self.parse_losses(losses)
                mean_loss = (
                    self._outputs["total_loss"]["sum"]
                    / self._outputs["total_loss"]["num"]
                )
                self._outputs.clear()
                self.accelerator.log(dict(eval_loss=mean_loss), step=self.cur_step)
                self.accelerator.print(
                    f"eval iter {self.cur_step}: mean loss {mean_loss} "
                )
                self.logger.info(f"eval iter {self.cur_step}: mean loss {mean_loss} ")
                release_memory()
                self.accelerator.wait_for_everyone()

    def forward_step(self, batch_dict):
        return self.model(batch_dict)

    def backward_step(self, loss):
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.max_grad_norm is not None:
            params = []
            for model in self.models:
                params += list(model.parameters())
            self.accelerator.clip_grad_norm_(
                params, self.max_grad_norm, self.grad_norm_type
            )
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if self.accelerator.sync_gradients and self.with_ema:
            for model, ema_model in zip(self.models, self.ema_models):
                if self.distributed_type == DistributedType.DEEPSPEED:
                    if (
                        self.accelerator.deepspeed_config["zero_optimization"]["stage"]
                        == 3
                    ):
                        state_dict = self.accelerator.get_state_dict(model)
                    else:
                        state_dict = self.accelerator.unwrap_model(model).state_dict()
                else:
                    state_dict = self.accelerator.unwrap_model(model).state_dict()
                ema_model.step(state_dict)

    def save_checkpoint_step(self):
        checkpoint_interval = int(
            min(self.checkpoint_interval, self.max_epochs * self.epoch_size)
        )
        if self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps:
            output_name = "checkpoint_epoch_{}_step_{}".format(
                self.cur_epoch, self.cur_step
            )
            output_dir = os.path.join(self.model_dir, output_name)
            if self.is_main_process:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                self.remove_checkpoint()
            release_memory()
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir)

    def save_model_hook(self, models, weights, output_dir):
        assert len(models) == 1
        model = self.accelerator.unwrap_model(models[0])
        if self.is_main_process:
            if len(weights) == 1:
                state_dict = weights.pop()
            elif (
                len(weights) == 0 and self.distributed_type == DistributedType.DEEPSPEED
            ):
                with torch.no_grad():
                    state_dict = self.accelerator.get_state_dict(models[0])
            else:
                assert False
            save_dtype = list(state_dict.values())[0].dtype
            if isinstance(model, ModuleDict):
                model_names = list(model.keys())
                for model_name in model_names:
                    model_output_dir = os.path.join(output_dir, model_name)
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model[model_name], "save_config"):
                        model[model_name].save_config(model_output_dir)
                    sub_state_dict = {
                        k[len(model_name) + 1 :]: v
                        for k, v in state_dict.items()
                        if k.startswith(model_name)
                    }
                    output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                    self.logger.info(f"Save {model_name} to {output_path}")
                    torch.save(sub_state_dict, output_path)
            else:
                model_name = getattr(self, "model_name", "model")
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                if hasattr(model, "save_config"):
                    model.save_config(model_output_dir)
                output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                self.logger.info(f"Save {model_name} to {output_path}")
                torch.save(state_dict, output_path)
        if self.with_ema:
            ema_model = self.ema_models[0]
            state_dict = ema_model.state_dict()
            if self.is_main_process:
                state_dict = utils.to_dtype(state_dict, save_dtype)
                save_config_path = os.path.join(output_dir, "ema_config.json")
                ema_model.save_config(save_config_path)
                if isinstance(model, ModuleDict):
                    model_names = list(model.keys())
                    for model_name in model_names:
                        model_output_dir = os.path.join(output_dir, model_name + "_ema")
                        os.makedirs(model_output_dir, exist_ok=True)
                        if hasattr(model[model_name], "save_config"):
                            model[model_name].save_config(model_output_dir)
                        sub_state_dict = {
                            k[len(model_name) + 1 :]: v
                            for k, v in state_dict.items()
                            if k.startswith(model_name)
                        }
                        output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                        self.logger.info(f"Save {model_name}_ema to {output_path}")
                        torch.save(sub_state_dict, output_path)
                else:
                    model_name = getattr(self, "model_name", "model")
                    model_output_dir = os.path.join(output_dir, model_name + "_ema")
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model, "save_config"):
                        model.save_config(model_output_dir)
                    output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                    self.logger.info(f"Save {model_name}_ema to {output_path}")
                    torch.save(state_dict, output_path)

    def load_model_hook(self, models, input_dir):
        assert len(models) == 0 or len(models) == 1
        if self.with_ema:
            model = self.models[0] if len(models) == 0 else models[0]
            model = self.accelerator.unwrap_model(model)
            ema_model = self.ema_models[0]
            config_path = os.path.join(input_dir, "ema_config.json")
            ema_model.load_config(config_path)
            if isinstance(model, ModuleDict):
                model_names = list(model.keys())
                state_dict = dict()
                for model_name in model_names:
                    input_path = os.path.join(
                        input_dir, model_name + "_ema", WEIGHTS_NAME
                    )
                    self.logger.info(f"Load {model_name}_ema from {input_path}")
                    sub_state_dict = torch.load(input_path, map_location="cpu")
                    sub_state_dict = {
                        model_name + "." + k: v for k, v in sub_state_dict.items()
                    }
                    state_dict.update(sub_state_dict)
            else:
                model_name = getattr(self, "model_name", "model")
                input_path = os.path.join(input_dir, model_name + "_ema", WEIGHTS_NAME)
                self.logger.info(f"Load {model_name}_ema from {input_path}")
                state_dict = torch.load(input_path, map_location="cpu")
            ema_model.load_state_dict(
                state_dict, device=self.device, dtype=torch.float32
            )
        if len(models) == 0:
            return
        model = models.pop()
        if isinstance(model, ModuleDict):
            model_names = list(model.keys())
            state_dict = dict()
            for model_name in model_names:
                input_path = os.path.join(input_dir, model_name, WEIGHTS_NAME)
                self.logger.info(f"Load {model_name} from {input_path}")
                sub_state_dict = torch.load(input_path, map_location="cpu")
                sub_state_dict = {
                    model_name + "." + k: v for k, v in sub_state_dict.items()
                }
                state_dict.update(sub_state_dict)
        else:
            model_name = getattr(self, "model_name", "model")
            input_path = os.path.join(input_dir, model_name, WEIGHTS_NAME)
            self.logger.info(f"Load {model_name} from {input_path}")
            state_dict = torch.load(input_path, map_location="cpu")
        model.load_state_dict(state_dict)

    def print_before_train(self):
        if not self.is_main_process:
            return
        for model in self.models:
            self.logger.info(model)
        msg = "num_processes: {}".format(self.num_processes)
        msg += ", process_index: {}".format(self.process_index)
        msg += ", train data_size: {}".format(self.train_data_size)
        msg += ", train batch_size: {}".format(self.batch_size)
        msg += ", train epoch_size: {}".format(self.epoch_size)
        msg += ", eval data_size: {}".format(self.eval_data_size)
        self.logger.info(msg)
        self.logger.warning(f"len(_train_dataloaders) = {len(self._train_dataloaders)}")
        self.logger.warning(f"len(train_dataloader) = {len(self.train_dataloader)}")
        self._epoch_tic = self._step_tic = self._start_tic = time.time()

    def print_step(self):
        if not self.is_main_process:
            return
        if self.cur_step % self.log_interval == 0:
            outputs = dict()
            for key, val in self._outputs.items():
                val = val["sum"] / val["num"] if val["num"] > 0 else float("nan")
                outputs[key] = val
            self._outputs.clear()
            self.accelerator.log(outputs, self.cur_step)
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(
                0, time_cost / self.log_interval * (self.max_steps - self.cur_step)
            )
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            lr = self.scheduler.get_last_lr()[0]
            if self._by_epoch:
                inner_step = (self.cur_step - 1) % self.epoch_size + 1
                msg = "Epoch[%d/%d][%d/%d]" % (
                    self.cur_epoch,
                    self.max_epochs,
                    inner_step,
                    self.epoch_size,
                )
            else:
                msg = "Step[%d/%d]" % (self.cur_step, self.max_steps)
            msg += " eta: %s, time: %.3f, data time: %.3f, speed: %.3f, lr: %.3e" % (
                eta_str,
                time_cost,
                self._data_time,
                speed,
                lr,
            )
            if self.mixed_precision == "fp16":
                if (
                    self.accelerator.scaler is not None
                    and self.accelerator.scaler.is_enabled()
                ):
                    grad_scale = self.accelerator.scaler.get_scale()
                elif self.distributed_type == DistributedType.DEEPSPEED:
                    optimizer = self.optimizer
                    if hasattr(optimizer, "optimizer"):
                        optimizer = optimizer.optimizer
                    if hasattr(optimizer, "loss_scaler"):
                        grad_scale = optimizer.loss_scaler.cur_scale
                    elif hasattr(optimizer, "cur_scale"):
                        grad_scale = optimizer.cur_scale
                    else:
                        assert False
                else:
                    grad_scale = None
            else:
                grad_scale = None
            if grad_scale is not None:
                msg += ", grad_scale: %.3f" % grad_scale
            for key, val in outputs.items():
                msg += ", %s: %.3f" % (key, val)
            self.logger.info(msg)
        if self._by_epoch and self.cur_step % self.epoch_size == 0:
            time_cost = time.time() - self._epoch_tic
            time_cost = str(datetime.timedelta(seconds=int(time_cost)))
            self._epoch_tic = time.time()
            self.logger.info("Total_time: %s" % time_cost)

    def print_after_train(self):
        if not self.is_main_process:
            return
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info("Total_time: %s" % time_cost)

    def parse_losses(self, losses):
        outputs = {}
        if isinstance(losses, dict):
            assert "total_loss" not in losses
            for key, val in losses.items():
                losses[key] = val.mean()
            loss = sum(losses.values())
            for key, val in losses.items():
                outputs[key] = self.accelerator.gather(val).mean()
            total_loss = sum(outputs.values())
            total_loss = self.accelerator.gather(total_loss).mean()
            outputs["total_loss"] = total_loss
        elif isinstance(losses, torch.Tensor):
            loss = losses.mean()
            total_loss = self.accelerator.gather(loss).mean()
            outputs["total_loss"] = total_loss
        else:
            assert False
        if self.loss_nan_total_limit > 0 and torch.isnan(loss).any():
            self._loss_nan_count += 1
            if self._loss_nan_count > self.loss_nan_total_limit:
                exit(-1)
        else:
            self._loss_nan_count = 0
        for key, val in outputs.items():
            if key not in self._outputs:
                self._outputs[key] = {"sum": 0.0, "num": 0}
            self._outputs[key]["sum"] += val.item()
            self._outputs[key]["num"] += 1
        return loss
