import os

# Set the API token as an environment variable
os.environ["HF_TOKEN"] = "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"


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_ckpt = "shng2025/gptesla"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config_small = AutoConfig.from_pretrained(
    "gpt2", vocab_size=len(tokenizer), gradient_checkpointing=True
)
model_small = AutoModelForCausalLM.from_config(config_small)


def model_size(model):
    return sum(t.numel() for t in model.parameters())


print(f"GPT-2 size: {model_size(model_small)/1000**2:.1f}M parameters")


# model_small.save_pretrained("models/" + model_ckpt + "-small", push_to_hub=True)


"""
dir_name = "hf_model_dir"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print(f"Directory '{dir_name}' was created.")
else:
    print(f"Directory '{dir_name}' already exists.")
"""

import torch
from torch.utils.data import IterableDataset


class ConstantLengthDataset(IterableDataset):

    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m = f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    print(m)
                    break
                try:
                    m = f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    print(m)
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    # iterator = iter(self.dataset)
                    more_examples = False
                    break

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs["input_ids"]:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


# Testing the Dataloader
from datasets import load_dataset, DownloadConfig

dataset = load_dataset("shng2025/gptesla-train", split="train", streaming=True)

shuffled_dataset = dataset.shuffle(buffer_size=100)
constant_length_dataset = ConstantLengthDataset(
    tokenizer, shuffled_dataset, num_of_sequences=10
)
dataset_iterator = iter(constant_length_dataset)

lengths = [len(b) for _, b in zip(range(5), dataset_iterator)]
print(f"Lengths of the sequences: {lengths}")


from argparse import Namespace

# GPTesla - 111M param setup in comment. Modification to make lighter training requirement needed
config = {
    "train_batch_size": 12,  # 12
    "valid_batch_size": 12,  # 12
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 5e-4,  # 5e-4
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 100,  # 2000
    "gradient_accumulation_steps": 1,  # 1
    "max_train_steps": 2000,  # 150000
    "max_eval_steps": 10,
    "seq_length": 1024,
    "seed": 1,
    "save_checkpoint_steps": 500,
}  # 15000

args = Namespace(**config)


from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
from huggingface_hub import Repository
import datasets, transformers


def setup_logging(project_name):
    logger = logging.getLogger(__name__)

    # setting up log directory
    import os

    dir_name = "./log"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' was created.")
    else:
        print(f"Directory '{dir_name}' already exists.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"./log/debug_{accelerator.process_index}.log"),
            logging.StreamHandler(),
        ],
    )
    if accelerator.is_main_process:  # We only want to set up logging once
        wandb.init(project=project_name, config=args)
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


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


from torch.utils.data.dataloader import DataLoader


def create_dataloaders(dataset_name):
    train_data = load_dataset(dataset_name + "-train", split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = load_dataset(
        dataset_name + "-valid", split="validation", streaming=True
    )

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, seq_length=args.seq_length
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, seq_length=args.seq_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


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


from transformers import set_seed
from accelerate import Accelerator

set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
project_name = "shng2025/gptesla-small"
logger, tb_writer, run_name = setup_logging(project_name.split("/")[1])
logger.info(accelerator.state)

# Load model and tokenizer
if accelerator.is_main_process:
    hf_repo = Repository("./", clone_from=project_name, revision=run_name)
model = AutoModelForCausalLM.from_pretrained("./", gradient_checkpointing=True)
tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Prepare the optimizer and learning rate scheduler
dataset_name = "shng2025/gptesla"
optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)


def get_lr():
    return optimizer.param_groups[0]["lr"]


# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
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
            unwrapped_model.save_pretrained("./")
            hf_repo.push_to_hub(
                commit_message=f"step {step}"
            )  # IsADirectoryError: [Errno 21] Is a directory: '/dli/hf_model_dir/./wandb/latest-run'
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
