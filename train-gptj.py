import os
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs # DummyOptim, DummyScheduler, 
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_scheduler
)
from peft import get_peft_model, LoraConfig, TaskType

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import argparse
from utils import load_yaml, format_metrics
from datetime import timedelta
from data_gptj import load_data

    
def evaluate(accelerator, model, val_dataloader, config ):
    model.eval()
    val_clm_loss = MeanMetric().to(model.device)

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(val_dataloader),
        ):
            if i == config["train_args"]["eval_steps"] // 10:
                break
                
            loss = model(**batch).loss

            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})

            val_clm_loss.update(loss_values["loss"])

    return val_clm_loss

    

def train(config):
    set_seed(config["train_args"]["seed"])
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=100_800))
    accelerator = Accelerator(log_with="wandb", 
                              kwargs_handlers=[timeout])
    accelerator.init_trackers(
            project_name=config["train_args"]["project_name"],
            config=config,
            init_kwargs={"wandb": {"entity": config["train_args"]["entity"]}},
        )
    accelerator.print(f"NUM GPUS: {accelerator.num_processes}")
    accelerator.print(config)


    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # case in LLaMa where there is no pad or eos token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    with accelerator.main_process_first():
        train_dataloader, val_dataloader = load_data(config, tokenizer, streaming=config["train_args"]["streaming"])

    checkpoint = config["train_args"]["gradient_checkpointing"]
    if "flash_attn" in config["train_args"]:
        model = AutoModelForCausalLM.from_pretrained(config["model_path"], 
                                                    gradient_checkpointing=checkpoint, 
                                                    use_cache=False if checkpoint else True,
                                                    flash_attn=config["train_args"]["flash_attn"],
                                                    trust_remote_code=True)
    else:
       model = AutoModelForCausalLM.from_pretrained(config["model_path"], 
                                                    gradient_checkpointing=checkpoint, 
                                                    use_cache=False if checkpoint else True,
                                                    trust_remote_code=True) 

    if config["use_lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        # if accelerator.state.deepspeed_plugin is None
        # or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        # else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"])

    total_steps = 0
    max_steps = config["train_args"]["max_steps"] // (accelerator.num_processes * config["train_args"]["per_device_train_batch_size"])
    remainder = config["train_args"]["max_steps"] % (accelerator.num_processes * config["train_args"]["per_device_train_batch_size"])

    if remainder > 0:
        max_steps += 1

    accelerator.print(f"Training for {max_steps} steps")

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    # if (
    #     accelerator.state.deepspeed_plugin is None
    #     or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    # ):
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=config["train_args"]["max_steps"],
    )
    # else:
    #     # it seems like deepspeed lr schedules divide by world size for the total number of steps
    #     # but not for warmup_num_steps
    #     scheduler = DummyScheduler(
    #         optimizer, total_num_steps=max_steps * accelerator.num_processes, warmup_num_steps=1000
    #     )

    accelerator.print(f"Training a {model.num_parameters():,} parameter model")

    device = accelerator.device
    model.to(device)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    accelerator.register_for_checkpointing(scheduler)

    train_clm_loss = MeanMetric().to(device)

    if config["train_args"]["resume_from_checkpoint"]:
        # Loads the DeepSpeed checkpoint from the specified path
        accelerator.print(f"Resumed from checkpoint: {config['train_args']['resume_from_checkpoint']}")
        accelerator.load_state(config["train_args"]["resume_from_checkpoint"])
        path = os.path.basename(config["train_args"]["resume_from_checkpoint"])
        training_difference = os.path.splitext(path)[0]

        resume_step = int(training_difference.replace("step_", ""))
    else:
        resume_step = -1

        
    accelerator.wait_for_everyone()

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    if config["train_args"]["resume_from_checkpoint"] and resume_step is not None:
        # We need to skip steps until we reach the resumed step
        train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        total_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    accelerator.print(f"Resuming training from step {resume_step}")

    while total_steps < max_steps:
        for step, batch in enumerate(train_dataloader):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            # check if nan or inf
            if torch.isnan(loss).any():
                accelerator.print(f"NAN in loss: {loss}")
                accelerator.print(f"batch: {batch}")
                accelerator.print(f"step: {step}")
                accelerator.print(f"lr: {optimizer.param_groups[0]['lr']}")

            if torch.isinf(loss).any():
                accelerator.print(f"inf in loss: {loss}")
                accelerator.print(f"batch: {batch}")
                accelerator.print(f"step: {step}")
                accelerator.print(f"lr: {optimizer.param_groups[0]['lr']}")

            accelerator.backward(loss)

            optimizer.step()

            if total_steps % (config["train_args"]["eval_steps"] // 10) == 0:
                accelerator.log({"lr": scheduler.get_last_lr()[0]}, step=total_steps)
            scheduler.step()

            total_steps += 1
            progress_bar.update(1)
            optimizer.zero_grad()

            # filter here
            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})
            train_clm_loss.update(loss_values["loss"])

            if total_steps > 0 and total_steps % config["train_args"]["save_steps"] == 0:
                accelerator.save_state(f"{config['train_args']['output_dir']}/step_{total_steps}")
                
            if total_steps > 0 and total_steps % config["train_args"]["eval_steps"] == 0:
                val_mlm_loss = evaluate(accelerator, model, val_dataloader, config)
                log_train = {
                    "train_mlm_loss": train_clm_loss.compute()
                }
                log_val = {
                    "val_mlm_loss": val_mlm_loss.compute()
                }

                accelerator.log({**log_train, **log_val}, step=total_steps)

                accelerator.print(format_metrics(log_train, "train", f" step {total_steps} "))
                accelerator.print(format_metrics(log_val, "val", f" step {total_steps} "))

                train_clm_loss.reset()

            if total_steps >= max_steps:
                break

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{config['train_args']['output_dir']}/final",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)

    train(config)
