                                                              

"""Pretrain utilities."""

import gc
import dataclasses
from datetime import datetime
import math
import logging
import os
import sys
from .log_handler import CustomHandler

                                                                                  
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time

                                             
_TRAIN_START_TIME = time.time()
import torch

                                                           
_grad_bucket_info_logged = False

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import set_logger, get_logger, log_first_rank

from t2v_flow.planner.schedule_pool import SchedulePool

from .utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
)
from .global_vars import (
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger,
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0("[" + string + "] datetime: {} ".format(time_str))


def _is_distributed_fatal_error(error):
    error_str = str(error).lower()
    return (
        "out of memory" in error_str
        or "connection" in error_str
        or "timeout" in error_str
        or "closed" in error_str
        or ("nccl" in error_str and ("error" in error_str or "failed" in error_str))
    )


def _record_resume_step(resume_iter):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank == 0:
        with open("oom_resume_step.txt", "w") as f:
            f.write(str(resume_iter))


def _abort_and_exit(resume_iter):
    _record_resume_step(resume_iter)
    if torch.distributed.is_initialized() and hasattr(torch.distributed, "abort"):
        try:
            torch.distributed.abort()
        except Exception:
            pass
    os._exit(100)


def num_floating_point_operations(args, batch_size):
                            
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
          
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
        12
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            + (args.num_query_groups / args.num_attention_heads)
            + (args.seq_length / args.hidden_size)
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )


def append_to_progress_log(string):
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, "a") as f:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = args.world_size
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                f"# GPUs: {num_gpus}\t{string}\n"
            )


def get_start_time_from_progress_log():
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    args = get_args()
    assert args.save is not None
    progress_log_filename = os.path.join(args.save, "progress.txt")

                                                               
                                                                                    
                                      
                                                                                     
                                                
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(": ")[1])

    with open(progress_log_filename, "r") as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split("\t")
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = _get_field(line_tokens[7], float)
            if world_size_in_line != args.world_size:
                                                                   
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = (
                        latest_num_floating_point_operations
                    )
    assert (
        start_time is not None and start_num_floating_point_operations is not None
    ), "Should have seen at least one 'Starting job' entry with same world_size"
    return (
        datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
        start_num_floating_point_operations,
    )


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

                                                                  
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )
                                            

                                


    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

                                                                    
    set_jit_fusion_options()

                                                               
                                                                
                         
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor(
        [_TRAIN_START_TIME], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")

    args = get_args()
    timers = get_timers()

    one_logger = get_one_logger()
    if one_logger:
        one_logger.log_metrics({"train_iterations_warmup": 5})
    set_logger(args)

                                          
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type
    )
                                                                                              
    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate " "scheduler are built")
    config = get_model_config(model[0])
                                                           
                 
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        )
    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")

                                                                                

                         
    print_rank_0("done with setup ...")
    timers.log(
        ["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"],
        barrier=True,
    )

    if not args.skip_train:
        print_rank_0("training ...")

        if args.dataloader_type == "cyclic" and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        try:
            if args.do_train and args.train_iters > 0:
                if args.debug:
                                          
                    log_first_rank("start training")
                iteration, num_floating_point_operations_so_far = train(
                    forward_step_func,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    train_data_iterator,
                    valid_data_iterator,
                    process_non_loss_data_func,
                    config,
                )
        except Exception as e:
            if _is_distributed_fatal_error(e):
                print(f"[fatal] Detected distributed error: {str(e)}")
                print("[fatal] Exiting with code 100 for auto-restart.")
                resume_iter = getattr(args, "curr_iteration", iteration)
                _abort_and_exit(resume_iter)
            else:
                print("[fatal] Non-distributed error, re-raising.")
                raise
        print_datetime("after training is done")

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
    else:
        print_rank_0("skipping training (--skip-train is on) ...")

        iteration = args.iteration

    if args.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )

    if args.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            iteration,
            process_non_loss_data_func,
            config,
            verbose=True,
            write_to_tensorboard=not args.skip_train,
        )
                                            
                                                 
                                                            


def update_train_iters(args):

                                                                
    if args.train_iters:
        return

                                                     
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
                                                       
        iterations = 0
        consumed_samples = 0
                       
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
               
        update_num_microbatches(0, consistency_check=False)
                        
                                                         
        iterations += (args.train_samples - consumed_samples) // args.global_batch_size
        args.train_iters = iterations

    print_rank_0("setting training iterations to {}".format(args.train_iters))


def get_model(
    model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True
):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

                  
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
                                                                              
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert (
                    args.pipeline_model_parallel_split_rank is not None
                ), "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder,
            )
        else:
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]
                  
                                                      
                                                                       
                                                                         
                                                           
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

                                 
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum([p.nelement() for p in model_module.parameters()])
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )
    print("get_model()  move model to cuda device========")
    from my_utils import print_cuda_memory_gb

                     
                                                                 

    for model_module in model:
        model_module.cuda(torch.cuda.current_device())
                                                        
                      
    if args.fp16 or args.bf16:
        print(
            f"[Rank ]==================================enter fp16======================="
        )
        model = [Float16Module(model_module, args) for model_module in model]
                                                        

    if wrap_with_ddp:
        config = get_model_config(model[0])
        model = [
            DDP(
                config,
                model_chunk,
                data_parallel_group=mpu.get_data_parallel_group(
                    with_context_parallel=True
                ),
                expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
                accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
                overlap_grad_reduce=args.overlap_grad_reduce,
                use_distributed_optimizer=args.use_distributed_optimizer,
                                                                                             
                                                                 
                disable_bucketing=(model_chunk_idx > 0),
                check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
                force_bucketing=getattr(args, 'force_bucketing', False),
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

                                                                                    
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()
    torch.cuda.empty_cache()
    print_cuda_memory_gb("[+ empty cache]After DDP wrapping")
    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

                               
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
                            
    elif args.train_samples:
                                                                  
                                                                 
                                                                
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception("either train-iters or train-samples should be provided.")

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    no_wd_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()

    model = get_model(model_provider_func, model_type)
    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    optimizer = get_megatron_optimizer(
        config, model, no_wd_decay_cond, scale_lr_cond, lr_mult
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.load is not None or args.pretrained_checkpoint is not None:
        timers("load-checkpoint", log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler, strict=False
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

                                                
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, opt_param_scheduler


def train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config
):
    from my_utils import global_timer

    """Single training step."""
    args = get_args()
    timers = get_timers()
                                                           
                       
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
                                     
    optimizer.zero_grad()

                   
    forward_backward_func = get_forward_backward_func()
                                           
                                             
                                                    
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )
                                           
                                 
                                            
                          
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

                       
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

                        
                                                
                        
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()
                       
                                   

                     
                      
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

                           
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

                          
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

             
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                                           
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

                                            
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
                          
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
                         
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
                                          
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(
                    key, torch.tensor([0.0], dtype=torch.float, device="cuda")
                )
                + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

              
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

                           
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

                                
    if one_logger:
        job_name = os.environ.get("SLURM_JOB_NAME", None)
        current_app_tag = f"{job_name}_{batch_size}_{args.world_size}"
        one_logger.log_app_tag(current_app_tag)

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

                         
                                           
    if args.log_timers_to_tensorboard and (
        iteration % args.tensorboard_log_interval == 0
    ):
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log(
                {"samples vs steps": args.consumed_train_samples}, iteration
            )
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate", learning_rate, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar(
                    "decoupled-learning-rate", decoupled_learning_rate, iteration
                )
            writer.add_scalar(
                "learning-rate vs samples", learning_rate, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"learning-rate": learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size", batch_size, iteration)
            writer.add_scalar(
                "batch-size vs samples", batch_size, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"batch-size": batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(
                key + " vs samples", loss_dict[key], args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale vs samples", loss_scale, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"loss-scale": loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar("world-size", args.world_size, iteration)
            writer.add_scalar(
                "world-size vs samples", args.world_size, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"world-size": args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm vs samples", grad_norm, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"grad-norm": grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros vs samples", num_zeros_in_grad, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"num-zeros": num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm vs samples", params_norm, args.consumed_train_samples
            )
            if wandb_writer:
                wandb_writer.log({"params-norm": params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(
            moe_loss_scale,
            iteration,
            writer,
            wandb_writer,
            total_loss_dict,
            args.moe_per_layer_logging,
        )

    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size
        )
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar(
                    "iteration-time", elapsed_time_per_iteration, iteration
                )
            if wandb_writer:
                wandb_writer.log(
                    {"iteration-time": elapsed_time_per_iteration}, iteration
                )
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time_per_iteration * 1000.0
        )
        if args.log_throughput:
            log_string += f" throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |"
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar("throughput", throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({"throughput": throughput}, iteration)
        assert learning_rate is not None
                                                                                           
        log_string += " learning rate: {:.6E} |".format(learning_rate)
        if args.decoupled_lr is not None and (
            mpu.is_pipeline_first_stage(ignore_virtual=True)
            or mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            assert decoupled_learning_rate is not None
            log_string += " decoupled learning rate: {:.6E} |".format(
                decoupled_learning_rate
            )
        else:
            assert decoupled_learning_rate is None
        log_string += " global batch size: {:5d} |".format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.tensor(
                    [0.0], dtype=torch.float, device="cuda"
                )
        log_string += " loss scale: {:.1f} |".format(loss_scale)
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[nan_iters_key]
        )
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)

        if args.debug:
            log_first_rank(log_string)

        if report_memory_flag and learning_rate > 0.0:
                                                                       
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(
                    args, num_microbatches=num_microbatches, verbose=True
                )
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def compute_throughputs_and_append_to_progress_log(
    iteration, num_floating_point_operations_so_far
):
    args = get_args()
    if args.save is None:
        return

                             
                                                                                        
                                    
    global _TRAIN_START_TIME
    job_throughput = (
        num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
    ) / ((time.time() - _TRAIN_START_TIME) * 10**12 * args.world_size)

                                                                                
                                                                                        
                                                 
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (
        num_floating_point_operations_so_far - start_num_floating_point_operations
    ) / (elapsed_time * 10**12 * args.world_size)

    tokens_so_far = args.consumed_train_samples * args.seq_length

    append_to_progress_log(
        f"Saved checkpoint\tIteration: {iteration}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}"
    )


def save_checkpoint_and_time(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
):
    args = get_args()
    timers = get_timers()
                                                                        
    timers("save-checkpoint", log_level=0).start(barrier=True)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
    )
    timers("save-checkpoint").stop(barrier=True)
    timers.log(["save-checkpoint"])

    if args.log_progress:
        compute_throughputs_and_append_to_progress_log(
            iteration, num_floating_point_operations_so_far
        )


def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

                               
    write_args_to_tensorboard()

                                                  
    for model_module in model:
        model_module.train()

                    
    total_loss_dict = {}

                 
    iteration = args.iteration
                            
    resume_file = "oom_resume_step.txt"
    from my_utils import get_global_logger
    logger = get_global_logger()
    resume_curr_iter = -1
    resume_flag = 0                             

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    try:
        if rank == 0:
            if os.path.exists(resume_file):
                logger.info(f"\n[Auto-Resume] rank0 found OOM resume file: {resume_file}")
                with open(resume_file, "r") as f:
                    content = f.read().strip()
                if content.isdigit():
                    resume_curr_iter = int(content)
                    resume_flag = 1

        if torch.distributed.is_initialized():
            flag_tensor = torch.tensor([resume_flag], dtype=torch.int32, device="cuda")
            iter_tensor = torch.tensor([resume_curr_iter], dtype=torch.int32, device="cuda")

            torch.distributed.broadcast(flag_tensor, src=0)
            torch.distributed.broadcast(iter_tensor, src=0)

            resume_flag = flag_tensor.item()
            resume_curr_iter = iter_tensor.item()

        if resume_flag == 1 and resume_curr_iter >= 0:
            per_iter_times = int(os.environ.get("PER_ITERS", 1))
            new_curr_iter = resume_curr_iter + 1
            new_iteration = new_curr_iter * per_iter_times

            logger.info(
                f"[Rank {rank}] resuming from curr_iter {new_curr_iter} "
                f"(iteration {new_iteration})"
            )

            args.curr_iteration = new_iteration
            iteration = new_iteration

        else:
            logger.info(f"[Rank {rank}] no crash record found; starting normally.")

        if rank == 0 and resume_flag == 1:
            os.remove(resume_file)

    except Exception as e:
        logger.error(f"Auto-resume failed; starting normally: {e}")


    one_logger = get_one_logger()
    if one_logger:
        iteration_start = iteration
        train_samples_start = args.consumed_train_samples
        train_samples_target = args.train_samples
        one_logger.log_metrics(
            {
                "train_samples_start": args.consumed_train_samples,
                "train_iterations_start": iteration,
                "train_samples_target": train_samples_target,
                "train_iterations_target": args.train_iters,
            }
        )

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

                                       
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync for model_chunk in model
            ]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [
            lambda x: optimizer.finish_param_sync(model_index, x)
            for model_index in range(len(model))
        ]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers("interval-time", log_level=0).start(barrier=True)
    print_datetime("before the start of training step")
    report_memory_flag = True
    exit = False

    if args.manual_gc:
                                                                                    
                                                                         
        assert (
            args.manual_gc_interval >= 0
        ), "Manual garbage collection interval should be laerger than or equal to 0."
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def track_e2e_metrics():
                                                             
        if one_logger:
            train_duration = timers("interval-time").active_time()                   
            train_samples = args.consumed_train_samples - train_samples_start
            train_iterations = iteration - iteration_start
            train_iterations_time_msecs_avg = (
                train_duration * 1000.0
            ) / train_iterations
            if eval_iterations:
                validation_iterations_time_msecs_avg = (
                    eval_duration * 1000.0
                ) / eval_iterations
            else:
                validation_iterations_time_msecs_avg = None

            one_logger.log_metrics(
                {
                    "train_iterations_end": iteration,
                    "train_samples_end": args.consumed_train_samples,
                    "train_iterations": train_iterations,
                    "train_samples": train_samples,
                    "train_iterations_time_msecs_avg": train_iterations_time_msecs_avg,
                    "validation_iterations_time_msecs_avg": validation_iterations_time_msecs_avg,
                }
            )


    from t2v_flow.executor import DynamicForwardStepHandler

    


    from t2v_flow.planner import SchedulePool
    sche_pool = SchedulePool()
    LOOKAHEAD_WINDOW = 3
    handler = DynamicForwardStepHandler()

                                                                                      
                                       
                                                                      
                                  
                                 


    while iteration < args.train_iters:
        if (
            args.profile
            and iteration == args.profile_step_start
            and torch.distributed.get_rank() in args.profile_ranks
        ):
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        per_iter_times = int(os.environ.get("PER_ITERS"))
        test_sp = int(os.environ.get("TEST_SP"))
                                       
        PLANNER_PATH = None
        curr_iter = iteration
        if os.environ.get("ENABLE_PROFILE_DIT_LAYER") == "1" or os.environ.get("PROFILE_TASK_TYPES") != "":

            PLANNER_PATH = f"t2v_flow/planner/task_yamls/vae_tasks_{1 << ( (iteration + per_iter_times*test_sp) // per_iter_times)}.yml"
        elif os.environ.get("EXPERIMENT") == "1":
                                                                   
                           
                                
                                                         
                                                                               
                       
                delta_iter = int(os.environ.get('DELTA_ITER'))
                curr_iter = ( iteration + delta_iter )// per_iter_times
                if os.environ.get('MAX_FRAMES', "") != "":
                                                                                   
                    # Multi-node scaling runs (>16 GPUs) select their plan set under an
                    # extra {n_gpus} level, e.g. .../720p/129/32/genetic/ for a 4-node
                    # (32-GPU) run; the default 16-GPU (2-node) run omits this level.
                    _n_gpus = (torch.distributed.get_world_size()
                               if torch.distributed.is_initialized()
                               else int(os.environ.get('WORLD_SIZE', '16')))
                    _cluster_seg = f"{_n_gpus}/" if _n_gpus > 16 else ""
                    PLANNER_PATH = f"t2v_flow/planner/generated_schedules/{os.environ.get('MODEL_TYPE')}/{os.environ.get('RESOLUTION')}/{os.environ.get('MAX_FRAMES')}/{_cluster_seg}{os.environ.get('SCHEDULE_TYPE')}/schedule_{curr_iter}.yaml"
                else:
                    PLANNER_PATH = f"t2v_flow/planner/generated_schedules/{os.environ.get('MODEL_TYPE')}/{os.environ.get('RESOLUTION')}/{os.environ.get('SCHEDULE_TYPE')}/schedule_{curr_iter}.yaml"
                if not os.path.exists(PLANNER_PATH):
                    print(f"[Arachne] Plan set exhausted at iteration {iteration} "
                          f"(missing {PLANNER_PATH}); ending measurement run gracefully.")
                    sys.exit(0)
                                                          
                                                  
                                                                        
                                                                                                                                                                                                            
                             
                                                 
                                                                                         
                                                                                                                                                                                                      
                       
                                                                                                                                                                                                      
        elif os.environ.get("SIMULATE_DATA_ONLY") == "1" or os.environ.get("FIXED_SHAPE") == "1":
            PLANNER_PATH = f"t2v_flow/planner/task_yamls/vae_tasks_4.yml"
            curr_iter = iteration
        
                                                 
                                                     
        from my_utils import global_timer

        handler.update_and_setup_for_iteration(plan_filepath=PLANNER_PATH, curr_iter=curr_iter)
                                                                                         

        handler.logger.info(
            f"===============================================curr iteraion {curr_iter}=================================================== "
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        global_timer.start(f"iteration-{curr_iter}")
        

                                                                                      
                                                                                
                                                                                    
                                                             
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert (
                get_num_microbatches() > num_microbatches
            ), "number of microbatches should be increasing due to batch size rampup"
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)
        if os.environ.get("MEMORY_SNAPSHOT"):
            torch.cuda.memory._record_memory_history(max_entries=80000)
        args.curr_iteration = iteration
                                         

                                                 

                                                                   
                                           
                                             
                               
                                   
                                             
                                
        simulate_data = os.environ.get('SIMULATE_DATA_ONLY', '0')

        if simulate_data == '1':
            print("Running in [Data Simulation Mode]. No training will be performed.")
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = simulate_data_loading_step(
                data_iterator=train_data_iterator,
                current_iteration=iteration,
                handler=handler,
                log_filepath=f"simulation_data/simulation_log_{os.environ.get('MODEL_TYPE')}_{os.environ.get('RESOLUTION')}.txt",
                model=model,
                optimizer=optimizer,
                config=config,
                
            )
        else:
            print("Running in [Dynamic Training Mode].")
            fatal_error = torch.tensor(
                0,
                dtype=torch.int32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            try:
                loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = dynamic_train_step(
                    forward_step_func=handler.forward_step,
                    data_iterator=train_data_iterator,
                    model=model,
                    optimizer=optimizer,
                    opt_param_scheduler=opt_param_scheduler,
                    config=config,
                    handler=handler,
                )
            except Exception as e:
                if _is_distributed_fatal_error(e):
                    fatal_error.fill_(1)
                    print_rank_0(
                        f"Fatal distributed error at iteration {iteration}: {str(e)}"
                    )
                else:
                    raise
                                               
                                           
                                                                
               
                                        
                                                
                                             
                               
                                                            

        
                                                
        if os.environ.get("MEMORY_SNAPSHOT"):
            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            file_name = os.path.join(
                os.environ.get("PROF_SAVE_PATH"), f"memory_cp8_{time_str}.pkl"
            )
            torch.cuda.memory._dump_snapshot(file_name)
            torch.cuda.memory._record_memory_history(enabled=None)

                                                                
        torch.distributed.barrier()
                                                               
        global_timer.stop(f"iteration-{curr_iter}")
        
        global_timer.step()
        global_timer.next_iteration()

        if os.environ.get("ENABLE_PROFILE_DIT_LAYER") == "1" and ( iteration+1 ) % 20 == 0:
            global_timer.dump()
                                   
            patterns_to_process = [
                (r'backward_.*', 'backward', 'dit_backward_data'),          
                (r'layer_.*', 'layer', 'dit_forward_data'),                
                (r'forward_.*', 'forward', 'dit_forward_data')             
            ]


            iter_filter = lambda i: i > 0

            for pattern, name, target_subdir in patterns_to_process:
                print(f"\n{'='*40}")
                print(f"Generating report for pattern: '{pattern}'")
                print(f"{'='*40}")

                base_filename = f"{name}_report_bs_sp_4_all_iter_{iteration}.txt"
                
                output_path_with_subdir = os.path.join(target_subdir, base_filename)

                detailed_stats = global_timer.generate_report(
                    stage_pattern=pattern,
                    output_filename=output_path_with_subdir,
                    iteration_filter=iter_filter
                )

        iteration += 1
        batch_size = (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        args.consumed_train_samples += batch_size
        num_floating_point_operations_so_far += num_floating_point_operations(
            args, batch_size
        )

                  
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        if iteration % args.log_interval == 0:
            track_e2e_metrics()

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group["is_decoupled_lr"]:
                decoupled_learning_rate = param_group["lr"]
            else:
                learning_rate = param_group["lr"]
        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )
        mpu.reset_pre_stage()

                    
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(
                iteration, model, optimizer, opt_param_scheduler
            )

                    
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            timers("interval-time").stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.disable_pre_hook()
            if args.manual_gc and args.manual_gc_eval:
                                      
                gc.collect()
            prefix = "iteration {}".format(iteration)
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                False,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += args.eval_iters
            timers("eval-time").stop()
            if args.manual_gc and args.manual_gc_eval:
                                                                          
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                optimizer.enable_pre_hook()
            timers("interval-time", log_level=0).start(barrier=True)

                       
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                )
                print_datetime("exiting program after receiving SIGTERM.")
                exit = True
                break

        if args.save and args.save_interval and iteration % args.save_interval == 0:
            timers("interval-time").stop()
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )
            saved_checkpoint = True
            timers("interval-time", log_level=0).start(barrier=True)

                                   
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int,
                device="cuda",
            )
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(
                        iteration,
                        model,
                        optimizer,
                        opt_param_scheduler,
                        num_floating_point_operations_so_far,
                    )
                print_datetime("exiting program after {} minutes".format(train_time))
                exit = True
                break

                                     
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                )
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            exit = True
            break

        if (
            args.profile
            and iteration == args.profile_step_end
            and torch.distributed.get_rank() in args.profile_ranks
        ):
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:
            if (
                args.manual_gc_interval != 0
                and iteration % args.manual_gc_interval == 0
            ):
                gc.collect()

    track_e2e_metrics()

                                          
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

                                                                                     
    if args.use_distributed_optimizer and args.overlap_param_gather:
        optimizer.disable_pre_hook()

                                                                                            
    if exit:
        sys.exit()

    from my_utils import global_timer

                                                                                        

    global_timer.dump()
                           

                                                    
                                
                                      
                                      
       

    return iteration, num_floating_point_operations_so_far


import torch
import torch.distributed as dist
import os

def simulate_data_loading_step(
    data_iterator,
    handler,
    current_iteration,                 
    log_filepath="simulation_data/simulation_log_new.txt",               
    forward_step_func=None,
    model=None,
    optimizer=None,
    opt_param_scheduler=None,
    config=None,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n==================== Iteration {current_iteration} ====================\n")
    dist.barrier()

    log_line_for_current_rank = ""                      

    try:
        data_batch = handler.get_batch(data_iterator)
        
                   
        images_shape = data_batch['images'].shape
        h = images_shape[4]
        w = images_shape[3]
        if h == 1280:
            images_shape = [images_shape[0],images_shape[1],images_shape[2],h,w]
        shape_info = {
            key: value.shape 
            for key, value in data_batch.items() 
            if hasattr(value, 'shape')
        }
        shape_info['images'] = images_shape
        log_line_for_current_rank = f"[Iter {current_iteration} | Rank {rank}] Shapes: {shape_info}\n"

    except StopIteration:
        log_line_for_current_rank = f"[Iter {current_iteration} | Rank {rank}] Data iterator is exhausted.\n"
        handler.logger.error("Data iterator is exhausted.")
    
    gathered_logs = [None] * world_size if rank == 0 else None
    
    dist.gather_object(
        obj=log_line_for_current_rank,
        object_gather_list=gathered_logs,
        dst=0
    )

    if rank == 0:
        with open(log_filepath, 'a', encoding='utf-8') as f:
            if gathered_logs:
                for line in gathered_logs:
                    f.write(line)
    
    dist.barrier()
    
    loss_dict = {}
    skipped_iter = 0
    grad_norm = 0.0
    num_zeros_in_grad = 0
    
    return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad


def dynamic_train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    handler,
):
    """Dynamic train step function."""

    args = get_args()
    timers = get_timers()

    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    forward_backward_func = get_forward_backward_func()

    my_tasks = handler.get_my_tasks_for_this_iteration()
    ops_to_profile_by_task = {
        "DIT": {"nccl:all_to_all": "self_cuda_time_total"},
        "DiT": {"nccl:all_to_all": "self_cuda_time_total"},
        "VAE": {r"tile_encoder_\d+": "self_cuda_time_total"}
    }
    from my_utils import create_profiler_context

    

    for sub_step, task in enumerate(my_tasks):
        handler.set_active_task_by_index(sub_step)

        is_training_task = task.task_type in ["DiT", "FULL", "DIT"]
        task_group = mpu.get_custom_group(task.gpus)

        handler.logger.info(
            f"[Rank {torch.distributed.get_rank()}] > Sub-step {sub_step + 1}/{len(my_tasks)}: "
            f"Executing task '{task.name}', training={is_training_task}"
        )

        with mpu.use_custom_group(group=task_group):
            ops_profile = ops_to_profile_by_task.get(task.task_type, {})
            with create_profiler_context(
                current_task_type = task.task_type,
                logger=handler.logger,
                ops_to_analyze=ops_profile,
            ):
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=not is_training_task,
                )

        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

    
    from t2v_flow.planner.grad_sync_planner import sync_grads_with_plan, _get_bucket_grad_tensors
    from my_utils import global_timer

                                                              
    global _grad_bucket_info_logged
    if not _grad_bucket_info_logged:
        _bt = _get_bucket_grad_tensors(model[0])
        _n = len(_bt)
        _sizes = [g.numel() * g.element_size() / (1024 ** 2) for g in _bt]
        handler.logger.info(
            f"[GradSync] n_buckets={_n} "
            f"use_pipeline={_n > 1 and handler.grad_sync_plan and bool(handler.grad_sync_plan.reps)} "
            f"total_grad={sum(_sizes):.1f}MB "
            f"bucket_sizes_mb=[{', '.join(f'{s:.1f}' for s in _sizes[:5])}"
            f"{'...' if _n > 5 else ''}]"
        )
        _grad_bucket_info_logged = True

    _mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
    _max_mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)
    handler.logger.info(
        f"Before sync_grads: "
        f"allocated={_mem_before:.1f}MB, max_allocated={_max_mem_before:.1f}MB"
    )

    global_timer.start("sync_grad_plan_overhead")
    try:
        sync_grads_with_plan(
            ddp_model=model[0],                                
            grad_sync_plan=handler.grad_sync_plan,
            mpu=mpu,
        )
    finally:
        global_timer.stop("sync_grad_plan_overhead")

    _mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
    _max_mem_after = torch.cuda.max_memory_allocated() / (1024 ** 2)
    handler.logger.info(
        f"After sync_grads: "
        f"allocated={_mem_after:.1f}MB, max_allocated={_max_mem_after:.1f}MB, "
        f"delta={_mem_after - _mem_before:+.1f}MB"
    )

                       
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

                      
    if (
        getattr(args, "vision_pretraining", False)
        and args.vision_pretraining_type == "dino"
    ):
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

                          
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()
    skipped_iter = 0
    grad_norm = 1
    num_zeros_in_grad = 0
             
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                                           
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def evaluate(
    forward_step_func,
    data_iterator,
    model,
    process_non_loss_data_func,
    config,
    verbose=False,
):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers("evaluate", log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank

        compute_feature_bank(model)

                                                     
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

                                                                     
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // (
        args.micro_batch_size * args.data_parallel_size
    )

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f"Evaluating on {args.eval_iters * eval_batch_size} samples")
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f"Evaluating iter {iteration}/{args.eval_iters}")

            forward_backward_func = get_forward_backward_func()
                                                       
            config.timers = None
            loss_dicts = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=eval_num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
            )
            config.timers = get_timers()

                                 
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                                          
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = (
                            total_loss_dict.get(
                                key,
                                torch.tensor([0.0], dtype=torch.float, device="cuda"),
                            )
                            + loss_dict[key]
                        )

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int,
                    device="cuda",
                )
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX
                )
                done = done_cuda.item()
                if done:
                    print_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True,
            )

                                        
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

    timers("evaluate").stop()
    timers.log(["evaluate"])

    return total_loss_dict, collected_non_loss_data, False


def evaluate_and_print_results(
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    process_non_loss_data_func,
    config,
    verbose=False,
    write_to_tensorboard=True,
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
    )
                                     
    if timelimit:
        return
    string = " validation loss at {} | ".format(prefix)
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar(
                "{} validation".format(key), total_loss_dict[key].item(), iteration
            )
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                args.consumed_train_samples,
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, iteration)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key),
                    ppl,
                    args.consumed_train_samples,
                )
            if wandb_writer and is_last_rank():
                wandb_writer.log(
                    {"{} validation".format(key): total_loss_dict[key].item()},
                    iteration,
                )

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

                                         
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)


def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

                                                      
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert (
            args.train_samples is None
        ), "only backward compatiblity support for iteration-based training"
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (
                (args.iteration // args.eval_interval)
                * args.eval_iters
                * args.global_batch_size
            )

                                                        
    is_distributed = getattr(
        build_train_valid_test_datasets_provider, "is_distributed", False
    )

                                 
                                                                     
    if is_distributed or mpu.get_tensor_context_parallel_rank() == 0:
                         
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider
        )
                           
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples
        )
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples
            )
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

                                                                     
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long,
            device="cuda",
        )
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

                    
    train_dataloader, valid_dataloader, test_dataloader = (
        build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)
    )

                                                                             

                      
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
                                                                                               
            return iter(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
