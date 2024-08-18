
import torch, os, multiprocessing
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from peft.utils.other import fsdp_auto_wrap_policy
from accelerate import Accelerator

os.environ["WANDB_API_KEY"] = "1b142007b7970985fb95f5abc8c34bf04678571a"

accelerator = Accelerator()

set_seed(1234)

model_name = "meta-llama/Meta-Llama-3.1-8B"
ds = load_dataset("tatsu-lab/alpaca")


def process(row):
    row["text"] = row["text"]+"<|end_of_text|>"
    return row
ds = ds.map(
    process,
    num_proc= multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = 128004
tokenizer.padding_side = 'right'

# compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# attn_implementation = 'flash_attention_2' if torch.cuda.is_bf16_supported() else 'sdpa'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_dtype = torch.float16
attn_implementation = 'sdpa'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
          model_name, torch_dtype=compute_dtype, attn_implementation=attn_implementation
)
# model = model.to(device)
# Enable gradient computation for input embeddings during fine-tuning
# This allows the embedding layer to be updated, improving adaptation to new data
# we assign require_grad as True to the output variables of the embedding layer
# although we tie the weights of LM head and input embeddings, the LM head has gradients flowing because of the loss, hence we only assign grad=true for outputs of input embedding layer
# register forward hook executes after each forward call of a nn.Module(in this case the Embedding layer)
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
# Moves gradients from GPU to CPU when there is no enough space to perform a forward pass on GPU
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
# LoRA 
# W = base_weights + (lora_alpha/lora_rank)*(BA)
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)
output_dir = "./Llama3.1_8b_LoRA/"
training_arguments = SFTConfig(
        output_dir=output_dir ,
        eval_strategy="steps",
        evaluation_strategy="no",
        do_eval=False,
        optim="adamw_torch",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        log_level="debug",
        logging_steps=1,
        learning_rate=1e-4,
        bf16 = False,
        max_steps=50,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
        save_strategy="steps",
        save_steps=5,  # Save every 5 steps
        save_total_limit=3
)
trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
)
fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
trainer.train()
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(output_dir)

