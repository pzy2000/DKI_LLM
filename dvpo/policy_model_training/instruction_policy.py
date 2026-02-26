import os
import json
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, get_peft_model,PeftConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, LogitsProcessorList, BitsAndBytesConfig, AutoModelForSequenceClassification,AutoModelForCausalLM,get_scheduler,Adafactor

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
#from vas_trainer import VASTrainer
#from policy_config import Policy_Config
from policyv1_config import Policyv1Config
from policyv1_trainer import Policyv1Trainer
from trl.core import LengthSampler
import wandb
import huggingface_hub
'''
    
##lr=2e-6
accelerate launch --num_machines 1 --num_processes 8 --config_file examples/accelerate_configs/deepspeed_zero1.yaml \
    /cosmos/hch/value/trl_fix/llama_policy.py  \
    --model_name "your_model_path" \
    --tokenizer_name "your_model_path" \
    --value_model "value_model_path"  \
    --dataset_name "dataset_path" \
    --load_from_json True \
    --subset "data/rl" \
    --output_dir 'output_path' \
    --learning_rate 2e-6 \
    --output_max_length 640 \
    --batch_size 2 \
    --mini_batch_size 1 \
    --ppo_epochs 4 \
    --gradient_accumulation_steps 2 \
    --adafactor False \
    --early_stopping False \
    --target_kl 0.1 \
    --batched_gen True \
    --save_freq 50 \
    --seed 42 \
    --bf16 True \
    --init_kl_coef 0.2 \
    --adap_kl_ctrl True \
    --epochs 4 

'''


tqdm.pandas()
wandb.login(key='')
huggingface_hub.login('')

@dataclass
class ScriptArguments:
    """
   The name of the Casual LM model we wish to fine with PPO
   """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    value_model: Optional[str] = field(default="", metadata={"help": "the value model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.4e-6, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the learning rate scheduler type"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=200, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "the outpur dir"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    epochs: Optional[int] = field(default=4, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
    load_from_json: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load dataset from json file"},
    )
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Use peft train model"})
    dataset_sample_num:  Optional[int] = field(default=10000, metadata={"help": "dataset num"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

policy_config = Policyv1Config(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    output_dir=script_args.output_dir,
    log_with=script_args.log_with,
    mixed_precision = 'bf16' if script_args.bf16 else 'fp16',
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    tracker_project_name="policy_train",
    tracker_kwargs={"wandb": {"name": script_args.output_dir}}
)



def build_response_train_dataset(script_args, dataset_name=script_args.dataset_name):
    ds = load_dataset('json',data_files=dataset_name, split='train')

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    tokenizer.add_eos_token = True

    def tokenize(sample):
        #sample["text"] = sample["question"] + sample["response"]
        messages = [
            {"role":"user","content":sample["question"]}
                    ]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt = True)
        sample["query"] = tokenizer.encode(text)
        #sample["response"] = tokenizer.encode(sample["response"])[1:]
        #print('sample response')
        #print(sample["value_response"])
        #print('tokenizer sample response')
        #print(tokenizer.encode(sample["value_response"]))
        #print(tokenizer.encode(sample["value_response"])[1:])
        #print(sample["response_tensor"])
        #sample["rewards"] = sample["rewards"]
        return sample

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(lambda sample: len(sample["question"]) > 0)
    ds = ds.filter(lambda sample: len(sample["query"]) < script_args.input_max_length)
    ds = ds.shuffle(seed=42).select(range(script_args.dataset_sample_num))
    #ds = ds.filter(lambda sample: len(sample["query"]) < 512)
    ds.set_format(type="torch")
    return ds


'''
def prepare_dataset(dataset, tokenizer):
    def tokenize(element):
        outputs = tokenizer(
            element["prompt"],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        load_from_cache_file=False,
    )'''





dataset = build_response_train_dataset(script_args=script_args, dataset_name=script_args.dataset_name)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(script_args.seed)

# Now let's build the model, the reference model, and the tokenizer.

if script_args.use_peft:
    print('yes')
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name,
        #load_in_8bit=True,
        #device_map='auto'
        #peft_config=peft_config
    )

    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model, use_gradient_checkpointing=True)
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
    model.is_peft_model = True

else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name)



tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,padding_side= 'left')

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.

if tokenizer.pad_token is None:
    #mistral
    #tokenizer.pad_token = tokenizer.unk_token

    # llama 3
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_eos_token = True


value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.value_model,
    #device_map=Accelerator().device
)
value_model.requires_grad_(False)
value_model.eval()
'''
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=policy_config.learning_rate,
)

lr_scheduler = get_scheduler(
    name=script_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=script_args.steps,
)'''


# We then build the VASTrainer, passing the model, the reference model, the tokenizer
policy_trainer = Policyv1Trainer(policy_config,
                                 model= model,
                                 ref_model=None,
                                 tokenizer=tokenizer,
                                 value_model=value_model,
                                 dataset=dataset,
                                 data_collator=collator,
                                 #optimizer=optimizer,
                                 #lr_scheduler=lr_scheduler
                                 )

steps = len(policy_trainer.dataloader)*script_args.epochs
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=script_args.learning_rate,
)

lr_scheduler = get_scheduler(
    name=ScriptArguments.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=steps
)
policy_trainer.optimizer= optimizer
policy_trainer.lr_scheduler = lr_scheduler

value_model.to(policy_trainer.accelerator.device)
#value_model = policy_trainer.accelerator.prepare(value_model)


# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the VASTrainer.
device = policy_trainer.accelerator.device
if policy_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch_id in range(script_args.epochs):

    for step, batch in enumerate(tqdm(policy_trainer.dataloader),start=0):

        question_tensors = batch["query"]


        response_tensors = policy_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # Compute reward score (using the sentiment analysis pipeline)

        #try:
        stats = policy_trainer.step(question_tensors, response_tensors)
        policy_trainer.log_stats(stats, batch)
        #except:
            #print(f"Error at step {step} when stepping")
            #continue

        if step !=0 and step % script_args.save_freq == 0:
            policy_trainer.save_pretrained(script_args.output_dir + f"epoch_{epoch_id}_step_{step}")
            print(f"epoch {epoch_id}, step {step}")



