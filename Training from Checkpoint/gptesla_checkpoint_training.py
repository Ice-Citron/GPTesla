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

# need a "Continue logging function"
"""
def setup_logging(project_name):
    logger = logging.getLogger(__name__)

    dir_name = "./log"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' was created.")
    else:
        print(f"Directory '{dir_name}' already exists.")

    # setting up log directory
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"log/debug_{accelerator.process_index}.log"),
            logging.StreamHandler(),
        ],
    )
    if accelerator.is_main_process:  # We only want to set up logging once
        wandb.init(project=project_name, config=args, dir="./../")
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {"0": 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer, run_name
"""


def checkpoint_state():

    checkpoint = {
        "lr_scheduler": lr_scheduler.state_dict(),
        "completed_steps": completed_steps,
        "logger": logger,
        "tb_writer": tb_writer,
        "run_name": run_name,
    }
    torch.save(checkpoint, f"checkpoint_{step}.pth")
    # Use Accelerate's built-in method if it covers all needs
    accelerator.save_state(output_dir="my_checkpoint")


def continue_logging(project_name, run_name, logger):
    # for tb_writer and logger will just use the states loaded from torch.save()
    # will only try to resume wandb

    wandb.resume(run_name=run_name)

    if accelerator.is_main_process:  # We only want to set up logging once
        wandb.init(project=project_name, config=args, dir="./../")
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

# Logging
logger, tb_writer, run_name = "SETUP NEEDED"
logger.info(accelerator.state)

# Load model and tokenizer
if accelerator.is_main_process:
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)

model = AutoModelForCausalLM.from_pretrained("./")  # , gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Load optimizer and learning rate scheduler
optimizer = ""  # likely no longer needed
lr_scheduler = ""


def get_lr():
    return optimizer.param_groups[0]["lr"]


"""
- completed_steps
"""

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


# advancing dataloader to correct position
completed_steps = ""

for i, _ in enumerate(train_dataloader):
    if i >= completed_steps - 1:
        break
for i, _ in enumerate(eval_dataloader):
    if i >= (completed_steps // args.save_checkpoint_steps) * args.max_eval_steps - 1:
        break

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
            # print(model.state_dict())
            # print(optimizer.state_dict())
            # print(steps)
            # print(train_dataloader.state_dict())
            # print(eval_dataloader.state_dict())
            worker_info = torch.utils.data.get_worker_info()
            print(worker_info)
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
