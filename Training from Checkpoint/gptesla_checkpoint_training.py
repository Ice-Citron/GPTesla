import os

import datasets, transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.optimization import get_scheduler
from datasets import load_dataset, DownloadConfig

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

import logging
import wandb
from huggingface_hub import Repository, create_branch
from accelerate import Accelerator
from argparse import Namespace


# Set the API token as an environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_checkpoint_state(step):

    checkpoint = {
        "lr_scheduler": lr_scheduler.state_dict(),
        "completed_steps": completed_steps,
        "logger": logger,
        "tb_writer": tb_writer,
        "run_name": run_name,
        "optimizer": optimizer
    }
    torch.save(checkpoint, f"torch_checkpoint/checkpoint_{step}.pth")


def load_checkpoint_torch(step, lr_scheduler, completed_steps, logger, tb_writer, run_name, optimizer):

    checkpoint = torch.load(f"torch_checkpoint/checkpoint_{step}.pth")
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    completed_steps.load_state_dict(checkpoint["completed_steps"])
    logger.load_state_dict(checkpoint["logger"])
    tb_writer.load_state_dict(checkpoint["tb_writer"])
    run_name.load_state_dict(checkpoint["run_name"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return step, lr_scheduler, completed_steps, logger, tb_writer, run_name, optimizer


def continue_logging(project_name, run_name, logger):
    # for tb_writer and logger will just use the states loaded from torch.save()
    # will only try to resume wandb

    if accelerator.is_main_process:  # We only want to set up logging once
        #wandb.init(project=project_name, config=args, dir="./../")
        wandb.init(project=project_name, id=run_name, resume="must", config=args, dir='./../')
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    loss = torch.mean(torch.cat(losses))

    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))

    return loss.item(), perplexity.item()


# Accelerator
accelerator = Accelerator(dispatch_batches=True)
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

project_name = "shng2025/gptesla-small"
dataset_name = "shng2025/gptesla"

# GPTesla - 111M param setup in comment. Modification to make lighter training requirement needed
config = {
    "train_batch_size": 12,  # 12
    "valid_batch_size": 12,  # 12
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 5e-4,  # 5e-4
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 700,  # 2000
    "gradient_accumulation_steps": 1,  # 1
    "max_train_steps": 50000,  # 150000
    "max_eval_steps": 10,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 50,
}  # 15000

args = Namespace(**config, **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size
set_seed(args.seed)


# Loading torch checkpoint
current_step = int(input("latest_step"))
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)
completed_steps = 0
logger = logging.getLogger(__name__)
tb_writer = SummaryWriter()
run_name = ""
lr_scheduler, completed_steps, logger, tb_writer, run_name, optimizer = load_checkpoint_torch(current_step, lr_scheduler, completed_steps, logger, tb_writer, run_name, optimizer)

print(current_step)
print(lr_scheduler)
print(logger)
print(tb_writer)
print(run_name)


# Load model and tokenizer
if accelerator.is_main_process:
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)

model = AutoModelForCausalLM.from_pretrained("./")  # , gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

def get_lr():
    return optimizer.param_groups[0]["lr"]


"""
- completed_steps
"""


# advancing dataloader to correct position
for i, _ in enumerate(train_dataloader):
    if i >= completed_steps:
        break
for i, _ in enumerate(eval_dataloader):
    if i >= (completed_steps // args.save_checkpoint_steps) * args.max_eval_steps:
        break

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Train model
model.train()
for step, batch in enumerate(train_dataloader, start=completed_steps + 1):
    loss = model(batch, labels=batch).loss
    log_metrics(
        step,
        {
            "lr": get_lr(),
            "samples": step * samples_per_step,
            "steps": completed_steps,
            "loss/train": loss.item(),
        },
    )
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info("Evaluating and saving model checkpoint")
        eval_loss, perplexity = evaluate()
        log_metrics(step, {"loss/eval": eval_loss, "perplexity": perplexity})
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            save_checkpoint_state(step)
            unwrapped_model.save_pretrained("./")
            accelerator.save_state(output_dir="my_checkpoint")
            hf_repo.push_to_hub(commit_message=f"step {step}")
        model.train()
    if completed_steps >= args.max_train_steps:
        break


# Evaluate and save the last checkpoint
logger.info("Evaluating and saving model after training")
eval_loss, perplexity = evaluate()
log_metrics(step, {"loss/eval": eval_loss, "perplexity": perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_main_process:
    unwrapped_model.save_pretrained("./")
    hf_repo.push_to_hub(commit_message="final model")
