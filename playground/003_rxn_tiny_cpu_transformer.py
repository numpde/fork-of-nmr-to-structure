import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
import math
import os
import wandb
from twig import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

os.environ["WANDB_SILENT"] = "True"   # no wandb prints at all
os.environ["WANDB_CONSOLE"] = "off"   # don't capture my stdout

OUT_PATH = Path(__file__).absolute().with_suffix('')
OUT_PATH.mkdir(exist_ok=True)

def require_config(cfg, key):
    if key not in cfg:
        log.error(f"Missing required config parameter: {key}")
        raise KeyError(f"Missing required config parameter: {key}")
    return cfg[key]

# Load configuration from YAML file
# [config_path] = Path("../src/").glob("**/transformer_template.yaml")
[config_path] = Path("../src/").glob("**/*tiny_cpu.yaml")

log.info(f"Loading config from: {config_path}")

with config_path.open('r') as f:
    config_yaml = yaml.safe_load(f)

# Set seed
torch.manual_seed(require_config(config_yaml, "seed"))

# --- Define the Transformer Model ---
class TinyTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super(TinyTransformer, self).__init__()
        self.d_model = require_config(config_yaml, "word_vec_size")
        self.src_embedding = nn.Embedding(input_vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(output_vocab_size, self.d_model)
        self.transformer = Transformer(
            d_model=self.d_model,
            nhead=require_config(config_yaml, "heads"),
            num_encoder_layers=require_config(config_yaml, "layers"),
            num_decoder_layers=require_config(config_yaml, "layers"),
            dim_feedforward=require_config(config_yaml, "transformer_ff"),
            dropout=require_config(config_yaml, "dropout"),
            batch_first=True
        )
        self.fc_out = nn.Linear(self.d_model, output_vocab_size)
        if str(require_config(config_yaml, "param_init_glorot")).lower() == 'true':
            self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        output = self.transformer(
            src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return self.fc_out(output)

# --- Noam Learning Rate Scheduler ---
class NoamLR:
    def __init__(self, optimizer, model_size, warmup_steps, lr_factor):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.lr_factor = lr_factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.lr_factor * (self.model_size ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        log.debug(f"NoamLR updated learning rate to: {lr:.6f}")
        self.optimizer.step()

def save_checkpoint(model, step):
    save_model_dir = OUT_PATH / "checkpoints"
    save_model_dir.mkdir(exist_ok=True)
    checkpoint_path = save_model_dir / f"transformer_checkpoint_step_{step}.pt"
    torch.save(model.state_dict(), str(checkpoint_path))
    log.info(f"Checkpoint saved at step {step}: {checkpoint_path}")

def main():
    wandb.init(project="tiny-transformer", config=config_yaml, settings=wandb.Settings(), dir=str(OUT_PATH))

    # For demonstration, use default vocab sizes (adjust or load as needed)
    input_vocab_size = 100
    output_vocab_size = 100

    model_instance = TinyTransformer(input_vocab_size, output_vocab_size)
    model_instance = model_instance.to(device)

    # Use Adam solver with parameters from the config
    optimizer = optim.Adam(
        model_instance.parameters(),
        lr=require_config(config_yaml, "learning_rate"),
        betas=(require_config(config_yaml, "adam_beta1"), require_config(config_yaml, "adam_beta2")),
        eps=1e-9
    )
    scheduler = NoamLR(
        optimizer,
        model_instance.d_model,
        require_config(config_yaml, "warmup_steps"),
        require_config(config_yaml, "learning_rate")
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=require_config(config_yaml, "label_smoothing"))

    # For dummy data: assume a fixed number of examples and compute seq_len from batch tokens
    batch_tokens = require_config(config_yaml, "batch_size")
    num_examples = 32
    seq_len = batch_tokens // num_examples

    log.info("Starting training")
    log.debug(f"Training config: num_examples={num_examples}, seq_len={seq_len}, "
              f"train_steps={require_config(config_yaml, 'train_steps')}, accum_count={require_config(config_yaml, 'accum_count')}")

    accumulated_loss = 0.0
    step_count = 0
    for step in range(1, require_config(config_yaml, "train_steps") + 1):
        model_instance.train()
        optimizer.zero_grad()

        src = torch.randint(0, input_vocab_size, (num_examples, seq_len), device=device)
        tgt = torch.randint(0, output_vocab_size, (num_examples, seq_len), device=device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model_instance(src, tgt_input)
        loss = criterion(output.view(-1, output_vocab_size), tgt_output.reshape(-1))
        loss.backward()

        accumulated_loss += loss.item()
        step_count += 1

        if step_count % require_config(config_yaml, "accum_count") == 0:
            if require_config(config_yaml, "max_grad_norm") > 0:
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), require_config(config_yaml, "max_grad_norm"))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % require_config(config_yaml, "report_every") == 0:
            avg_loss = accumulated_loss / require_config(config_yaml, "accum_count")
            log.info(f"Step {step}, Average Loss: {avg_loss:.4f}")
            wandb.log({"step": step, "loss": avg_loss})
            accumulated_loss = 0.0

        if step % require_config(config_yaml, "save_checkpoint_steps") == 0:
            save_checkpoint(model_instance, step)

    log.info("Training completed")


if __name__ == "__main__":
    main()
