

import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, LogitsProcessorList, BitsAndBytesConfig, AutoModelForSequenceClassification,Adafactor,get_scheduler

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
from vas_trainer import VASTrainer
from vas_config import VASConfig
import wandb

tqdm.pandas()
wandb.login(key='')

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=True, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the max length of sequence"})

parser = HfArgumentParser((ScriptArguments, VASConfig))
args, vas_config = parser.parse_args_into_dataclasses()




trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

def build_response_train_dataset(config, dataset_name):
    ds = load_dataset('json',data_files=dataset_name, split='train')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.pad_token
    tokenizer.add_eos_token = True

    def process_preference(sample):
        question = sample["question"]
        chosen = sample["response_j"]
        reject = sample["response_k"]

        
        chosen_sample = {
            "text": question + chosen,
            "query": tokenizer.encode(question)[:-1],
            "response": tokenizer.encode(chosen)[1:],
            "rewards": 1.0
        }
        
        reject_sample = {
            "text": question + reject,
            "query": tokenizer.encode(question)[:-1],
            "response": tokenizer.encode(reject)[1:],
            "rewards": -1.0
        }
        return [chosen_sample, reject_sample]

    new_samples = []
    for sample in ds:
        new_samples.extend(process_preference(sample))

    # 构造新的数据集
    new_ds = Dataset.from_list(new_samples)

    new_ds = new_ds.filter(lambda sample: len(sample["response"]) > 0)
    new_ds = new_ds.filter(lambda sample: (len(sample["response"]) + len(sample["query"])) < args.max_length)
    print(len(new_ds))
    new_ds.set_format(type="torch")
    return new_ds


dataset = build_response_train_dataset(vas_config,dataset_name=vas_config.dataset_name)

#evaluate
#eval_dataset = build_response_train_dataset(vas_config,dataset_name=vas_config.eval_dataset_name)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(vas_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model = trl_model_class.from_pretrained(
    vas_config.model_name,
    #quantization_config=quantization_config,
    #trust_remote_code=args.trust_remote_code,
    #device_map='auto',
)

if args.use_peft:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model, use_gradient_checkpointing=True)
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
    model.is_peft_model = True
else:
    model.is_peft_model= False
torch.nn.init.zeros_(model.v_head.summary.weight)
torch.nn.init.zeros_(model.v_head.summary.bias)
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0

if not vas_config.train_value_only:
    ref_model = trl_model_class.from_pretrained(
         vas_config.ref_model_name,
        #quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        #device_map='auto',
    )
else:
    ref_model = None

tokenizer = ref_tokenizer =AutoTokenizer.from_pretrained(vas_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True





# We then build the VASTrainer, passing the model, the reference model, the tokenizer
vas_trainer = VASTrainer(vas_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator,validation_dataset =None,validation_data_collator=collator)
steps = len(vas_trainer.dataloader)*vas_config.epochs
optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=vas_config.learning_rate,
)

lr_scheduler = get_scheduler(
    name=vas_config.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=steps
)
vas_trainer.optimizer= optimizer
vas_trainer.lr_scheduler = lr_scheduler
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the VASTrainer.
device = vas_trainer.accelerator.device
if vas_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

#reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
#reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(vas_trainer.accelerator.device)
#reward_model = vas_trainer.accelerator.prepare(reward_model)
#reward_model.requires_grad_(False)
#reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
#reward_model.eval()


#vas_trainer.evaluate()



for step, batch in enumerate(tqdm(vas_trainer.dataloader)):
    #print(batch)
    query_tensors = batch["query"]
    response_tensors = batch["response"]
    #print(query_tensors)
    #print(response_tensors)
    #print('query_tensor_size')
    #print(query_tensors[0].size())
    #print('response_tensor_size')
    #print(response_tensors[0].size())

    # Compute score
    #texts = batch["text"]
    rewards = batch["rewards"]
    #print(rewards)
    #for text in texts:
        #inputs_ids = reward_tokenizer.encode(text, return_tensors='pt').to(reward_model.device)
        #reward_outputs = reward_model(inputs_ids)
        #reward = reward_outputs.logits[0]
        #rewards.append(reward.squeeze())

    # Run VAS step
    #print('rewards')
    #print(rewards)
    stats = vas_trainer.step(query_tensors, response_tensors, rewards)
    vas_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

    #save checkpoint
    if step !=0 and step % vas_config.save_freq == 0:
        vas_trainer.save_pretrained(vas_config.output_dir + f"/step_{step}")
        print(f"step {step}")

vas_trainer.save_pretrained(vas_config.output_dir)

# Decoding example
#query = "Human: How are you doing today? Assistant:"
#inputs = ref_tokenizer.encode(query, return_tensors='pt').to(reward_model.device)
#output = vas_trainer.generate(inputs, vas_generation=True, beta=3.0)

