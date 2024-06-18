---
title: Finetuning LLM
date: 2023-12-01 00:00:00 +0800
categories: [NLP, Transformer, Finetuning_LLM]
tags: [finetuningllm]
---

#

<div align="center">
  <img src="https://media.licdn.com/dms/image/D5622AQHHpcUgitQuEg/feedshare-shrink_800/0/1717144084301?e=1721865600&v=beta&t=irZqkr64joJhF4YJZ8lzR1pTfaO7w033JZX7fwq2uPE" alt="gd" width="600" height="800" />

</div>
## Instruct Finetuning

### Datasets

```python

# Some datasets for you to try

#pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)

pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)

instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = datasets.load_dataset(finetuning_dataset_path)
print(finetuning_dataset)

taylor_swift_dataset = "lamini/taylor_swift"
bts_dataset = "lamini/bts"
open_llms = "lamini/open_llms"
dataset_swiftie = datasets.load_dataset(taylor_swift_dataset)
print(dataset_swiftie["train"][1])
dataset_swiftie = datasets.load_dataset(taylor_swift_dataset)
print(dataset_swiftie["train"][1])



```

### Data preparation

Final Function

```python
# Tokenize the instruction dataset
def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]
​
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )
​
    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )
    return tokenized_inputs

finetuning_dataset_loaded = datasets.load_dataset("json", data_files=filename, split="train")
​
tokenized_dataset = finetuning_dataset_loaded.map(
    tokenize_function,
    batched=True,
    batch_size=1,
    drop_last_batch=True
)
print(tokenized_dataset)

tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

# Prepare test/train splits
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
print(split_dataset)
```

```python
# Example
import pandas as pd
import datasets
​
from pprint import pprint
from transformers import AutoTokenizer
# Tokenizing text
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

text = "Hi, how are you?"
encoded_text = tokenizer(text)["input_ids"]
encoded_text

decoded_text = tokenizer.decode(encoded_text)
print("Decoded tokens back into text: ", decoded_text)

# Tokenize multiple texts at once
list_texts = ["Hi, how are you?", "I'm good", "Yes"]
encoded_texts = tokenizer(list_texts)
print("Encoded several texts: ", encoded_texts["input_ids"])

# Padding and truncation
tokenizer.pad_token = tokenizer.eos_token
encoded_texts_longest = tokenizer(list_texts, padding=True)
print("Using padding: ", encoded_texts_longest["input_ids"])
encoded_texts_truncation = tokenizer(list_texts, max_length=3, truncation=True)
print("Using truncation: ", encoded_texts_truncation["input_ids"])
tokenizer.truncation_side = "left"
encoded_texts_truncation_left = tokenizer(list_texts, max_length=3, truncation=True)
print("Using left-side truncation: ", encoded_texts_truncation_left["input_ids"])
encoded_texts_both = tokenizer(list_texts, max_length=3, truncation=True, padding=True)
print("Using both padding and truncation: ", encoded_texts_both["input_ids"])
```

### Prepare instruction dataset

```python
import pandas as pd
​
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()
​
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]
​
prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
​
num_examples = len(examples["question"])
finetuning_dataset = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append({"question": text_with_prompt_template, "answer": answer})
​
processed_data = []
for j in top_m:
  if not j["input"]:
    processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
  else:
    processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])

  processed_data.append({"input": processed_prompt, "output": j["output"]})

from pprint import pprint
print("One datapoint in the finetuning dataset:")
pprint(finetuning_dataset[0])

```

### Prompt templates

```python
# Prompt templates
prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

prompt_template_q = """### Question:
{question}

### Answer:"""

prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""
```

### Training

Technically, it's only a few lines of code to run on GPUs (elsewhere, ie. on Lamini).

- Choose base model.
- Load data.
- Train it. Returns a model ID, dashboard, and playground interface.

```python
from llama import BasicModelRunner

model = BasicModelRunner("EleutherAI/pythia-410m")
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True)
```

```python
# Finetune a model in 3 lines of code using Lamini
model = BasicModelRunner("EleutherAI/pythia-410m")
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True)
out = model.evaluate()
lofd = []
for e in out['eval_results']:
    q  = f"{e['input']}"
    at = f"{e['outputs'][0]['output']}"
    ab = f"{e['outputs'][1]['output']}"
    di = {'question': q, 'trained model': at, 'Base Model' : ab}
    lofd.append(di)
df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-align': 'left'})
style_df = style_df.set_properties(**{"vertical-align": "text-top"})
style_df
```

This is the open core of Lamini's llama library

```python

import os
import lamini
​
lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines
​
from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner
​
​
logger = logging.getLogger(__name__)
global_config = None

#Load the Lamini docs dataset
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
# From Huggingface
dataset_path = "lamini/lamini_docs"
use_hf = True

# Set up the model, training config, and tokenizer
model_name = "EleutherAI/pythia-70m"
training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)
​
print(train_dataset)
print(test_dataset)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")
base_model.to(device)

# Define function to carry out inference
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )
​
  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )
​
  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
​
  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]
​
  return generated_text_answer

# Try the base model
test_text = test_dataset[0]['question']
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))

# Setup training
max_steps = 3
trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name
training_args = TrainingArguments(
​
  # Learning rate
  learning_rate=1.0e-5,
​
  # Number of training epochs
  num_train_epochs=1,
​
  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
  max_steps=max_steps,
​
  # Batch size for training
  per_device_train_batch_size=1,
​
  # Directory to save model checkpoints
  output_dir=output_dir,
​
  # Other arguments
  overwrite_output_dir=False, # Overwrite the content of the output directory
  disable_tqdm=False, # Disable progress bars
  eval_steps=120, # Number of update steps between two evaluations
  save_steps=120, # After # steps model is saved
  warmup_steps=1, # Number of warmup steps for learning rate scheduler
  per_device_eval_batch_size=1, # Batch size for evaluation
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,
  gradient_checkpointing=False,
​
  # Parameters for early stopping
  load_best_model_at_end=True,
  save_total_limit=1,
  metric_for_best_model="eval_loss",
  greater_is_better=False
)
# floating point Operation
model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)
​
print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")

trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# Train a few steps
training_output = trainer.train()
# Save model locally
save_dir = f'{output_dir}/final'
​
trainer.save_model(save_dir)
print("Saved model to:", save_dir)
finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
​
finetuned_slightly_model.to(device)
​
# Run slightly trained model
test_question = test_dataset[0]['question']
print("Question input (test):", test_question)
​
print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))
test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)
```

```python
# Run same model trained for two epochs
finetuned_longer_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")
finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))

# Run much larger trained model and explore moderation
bigger_finetuned_model = BasicModelRunner(model_name_to_id["bigger_model_name"])
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output)

count = 0
for i in range(len(train_dataset)):
 if "keep the discussion relevant to Lamini" in train_dataset[i]["answer"]:
  print(i, train_dataset[i]["question"], train_dataset[i]["answer"])
  count += 1
print(count)
```

### Explore moderation

```python
# Explore moderation using small model
# First, try the non-finetuned base model:

base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))
# Now try moderation with finetuned small model
print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))

```

### Benchmark

- ARC It's a set of grade school questions.
- HellaSwag is a test of common sense.
- MMLU covers a lot of elementary school subjects,
- TruthfulQA measures the model's ability to reproduce falsehoods that you can commonly find online.

Error Analysis - Framework for analyzing and evaluating your model is called error analysis.

Categorize error

- misspelling
- Too Long
- Repetitive

- **ROUGE Metrics:**

  - ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation.
  - Evaluation of automatically generated summaries by comparing them to human-generated reference summaries.
  - Utilizes recall, precision, and F1 scores based on unigrams, bigrams, and longest common subsequence (LCS).
  - Consideration of ordering of words through bigrams and LCS.

- **ROUGE-1 Metric:**
  - Focuses on unigrams (single words) in the comparison.
  - Calculation involves recall, precision, and F1 score for unigram matches between reference and generated output.
- **ROUGE-2 Metric (Bigrams):**
  - Takes into account bigrams (pairs of words) for a more nuanced evaluation.
  - Scores are lower than ROUGE-1 as bigram matches are less likely in longer sentences.
- **ROUGE-L Metric (Longest Common Subsequence):**

  - Considers the longest common subsequence in both reference and generated output.
  - Calculates recall, precision, and F1 score based on the length of the longest common subsequence.
  - Addresses issues with simple ROUGE scores that may yield high scores for poor completions.

- **Issues with Simple ROUGE Scores:**

  - Potential for a bad completion to result in a good score.
  - Example: Repeating a word multiple times in the generated output.
  - Clipping function with modified precision helps limit unigram matches to the maximum count in the reference.

- **BLEU Score (Bilingual Evaluation Understudy):**

  - Evaluates the quality of machine-translated text.
  - Calculates average precision over multiple n-gram sizes (unigrams, bigrams, etc.).
  - Average precision is then averaged across all n-gram sizes to obtain the BLEU score.
  - Simple yet widely used for translation tasks.

- **Calculation of BLEU Score:**

  - Average precision is computed for each n-gram size.
  - These individual calculations are averaged to determine the final BLEU score.
  - Example: Candidate sentences evaluated against a human-generated reference sentence.

- **Use of ROUGE and BLEU:**
  - Both metrics provide automated, structured ways to measure the similarity of sentences.
  - Useful for diagnostic evaluation of summarization (ROUGE) and translation tasks (BLEU).
  - Should be used in conjunction with larger evaluation benchmarks for a comprehensive model assessment.

### Evaluation

Technically, there are very few steps to run it on GPUs, elsewhere (ie. on Lamini).

```python
finetuned_model = BasicModelRunner(
    "lamini/lamini_docs_finetuned"
)
finetuned_output = finetuned_model(
    test_dataset_list # batched!
)
```

Let's look again under the hood! This is the open core code of Lamini's llama library.

```python
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd
​
import transformers
import datasets
import torch
​
from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM
​
logger = logging.getLogger(__name__)
global_config = None
dataset = datasets.load_dataset("lamini/lamini_docs")
​
test_dataset = dataset["test"]
print(test_dataset[0]["question"])
print(test_dataset[0]["answer"])

# Load finetuned model from HF
model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Setup a really basic evaluation function
def is_exact_match(a, b):
    return a.strip() == b.strip()

# dropout is disabled
model.eval()

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.encode(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens
  )
​
  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )
​
  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
​
  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]
​
  return generated_text_answer

# Run model and compare to expected answer
test_question = test_dataset[0]["question"]
generated_answer = inference(test_question, model, tokenizer)
print(test_question)
print(generated_answer)

answer = test_dataset[0]["answer"]
print(answer)
exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

# Run over entire dataset
n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']
​
    try:
      predicted_answer = inference(question, model, tokenizer)
    except:
      continue
    predictions.append([predicted_answer, answer])
​
    #fixed: exact_match = is_exact_match(generated_answer, answer)
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)
​
    if i > n and n != -1:
      break
print('Number of exact matches: ', sum(metrics['exact_matches']))
df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)

# Evaluate all the data
evaluation_dataset_path = "lamini/lamini_docs_evaluation"
evaluation_dataset = datasets.load_dataset(evaluation_dataset_path)
pd.DataFrame(evaluation_dataset)

#Try the ARC benchmark
#This can take several minutes

!python lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=lamini/lamini_docs_finetuned --tasks arc_easy --device cpu

```

###

```python
import jsonlines
import itertools
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset


# Data visualize
n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)

examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text

# Load data from jsonl
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df
# Save data in jsonl
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

# Preprocess data
if "question" in examples and "answer" in examples:
  text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
  text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
  text = examples["input"][0] + examples["output"][0]
else:
  text = examples["text"][0]


# Hydrate prompts (add data to prompts)
text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template

num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

pprint(finetuning_dataset_text_only[0])
pprint(finetuning_dataset_question_answer[0])

##


finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)

# Compare non-instruction-tuned vs. instruction-tuned models
dataset_path_hf = "lamini/alpaca"
dataset_hf = load_dataset(dataset_path_hf)
print(dataset_hf)

non_instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
non_instruct_output = non_instruct_model("Tell me how to train my dog to sit")
print("Not instruction-tuned output (Llama 2 Base):", non_instruct_output)

instruct_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
instruct_output = instruct_model("Tell me how to train my dog to sit")
print("Instruction-tuned output (Llama 2): ", instruct_output)

## Try smaller models
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )
​
  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )
​
  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)
​
  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]
​
  return generated_text_answer

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
print(finetuning_dataset)

test_sample = finetuning_dataset["test"][0]
print(test_sample)
​
print(inference(test_sample["question"], model, tokenizer))

# Compare to finetuned small model
instruction_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
print(inference(test_sample["question"], instruction_model, tokenizer))

```

## Guarantee Valid JSON Output with Lamini

https://www.lamini.ai/blog/guarantee-valid-json-output-with-lamini

# LangChain for LLM Application Development

## LangChain: Memory

Outline

- ConversationBufferMemory

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)
```

- ConversationBufferWindowMemory

```python
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)
# saves k conv only
```

- ConversationTokenBufferMemory

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})
```

- ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryBufferMemory

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})
```

## Chains in LangChain

- LLMChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
```

- SimpleSequentialChain

```python
from langchain.chains import SimpleSequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)
​
# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
​
# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)
# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )
overall_simple_chain.run(product)
```

- SequentialChain

```python
from langchain.chains import SequentialChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)
​
# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt,
                     output_key="English_Review"
                    )
​
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt,
                     output_key="summary"
                    )
​
# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )
​
​
# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )
​
# overall_chain: input= Review
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
review = df.Review[5]
overall_chain(review)
```

- Router Chain

````python
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.
​
Here is a question:
{input}"""
​
​
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts,
answer the component parts, and then put them together\
to answer the broader question.
​
Here is a question:
{input}"""
​
history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.
​
Here is a question:
{input}"""
​
​
computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity.
​
Here is a question:
{input}"""
prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(temperature=0, model=llm_model)
​
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.
​
<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
\```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
\```
​
REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.
​
<< CANDIDATE PROMPTS >>
{destinations}
​
<< INPUT >>
{{input}}
​
<< OUTPUT (remember to include the ```json)>>"""
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
​
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True
                        )
chain.run("What is black body radiation?")
chain.run("what is 2 + 2")
chain.run("Why does every cell in our body contain DNA?")
````

## LangChain: Q&A over Documents

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
from langchain.indexes import VectorstoreIndexCreator
#pip install docarray
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
response = index.query(query)
display(Markdown(response))
Step By Step
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)
docs = loader.load()
docs[0]
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
len(docs)
docs[0]
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
​
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")
​
display(Markdown(response))
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
response = index.query(query, llm=llm)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
```

## LangChain: Evaluation

Outline:

- Example generation
- Manual evaluation (and debuging)
- LLM-assisted evaluation
- LangChain evaluation platform

```python

# Create our QandA application
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

#Coming up with test datapoints
data[10]
data[11]
Hard-coded examples
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

# LLM-Generated examples
from langchain.evaluation.qa import QAGenerateChain
​
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
# the warning below can be safely ignored
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
new_examples[0]
data[0]
Combine examples
examples += new_examples
qa.run(examples[0]["query"])
Manual Evaluation
import langchain
langchain.debug = True
qa.run(examples[0]["query"])
# Turn off the debug mode
langchain.debug = False

# LLM assisted evaluation
predictions = qa.apply(examples)
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
graded_outputs[0]
```

LangChain evaluation platform
The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/. Use the invite code lang_learners_2023

## LangChain: Agents

Outline:
Using built in LangChain tools: DuckDuckGo search and Wikipedia
Defining your own tools

```python

# Built-in LangChain tools
#!pip install -U wikipedia
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model=llm_model)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
agent("What is the 25% of 300?")

#Wikipedia example
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question)

# Python Agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff","Fusion"],
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")

# View detailed outputs of the chains
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")
langchain.debug=False
# Define your own tool
#!pip install DateTime
from langchain.agents import tool
from datetime import date
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
agent= initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

#Note: The agent will sometimes come to the wrong conclusion (agents are a work in progress!).
# If it does, please try running it again.

try:
    result = agent("whats the date today?")
except:
    print("exception on external access")
```

#

## RAG

Building end-to-end RAG pipelines has been made easy with LlamaIndex, check out a simple yet powerful approach with open-source embeddings from Hugging Face & LLM with Meta Llama3

📚 Learn about each step in building a search & retrieval system from data ingestion, chunking, embedding, creating vector store index (in-memory db), to query engine.

🔗 Google Colab notebook - https://lnkd.in/gSKXry6Q

https://huggingface.co/learn/cookbook/structured_generation

# positional encoding

<div align="center">
  <img src="https://media.licdn.com/dms/image/D5622AQGqaMuoKY2oJA/feedshare-shrink_800/0/1712247238635?e=1721865600&v=beta&t=fERJl95gefpSBwzms_rDt74vDbArUk6eRvkfObC7_x4" alt="gd" width="400" height="600" />

</div>

Did you know that LLama 2 is probably among the best choice if you need a large context window with an open-source model? In fact, any model using the RoPE positional embedding is a good bet!

4096 tokens, that's about 3000 words. Not bad but it limits the possible applications. The typical Transformer architecture is composed of Embeddings to encode the text input, multiple transformer blocks, and a prediction head specific to the learning task the LLM is used for. To encode the text, we use a text embedding matrix T that has the size of the token vocabulary and a positional embedding P that encodes the position of the token in the input sequence. That position embedding size defines the context size. That embedding can be learned or it can be a simple sin function of the position index. Typically they are added together T + P such that the same word is encoded differently at positions i and j.

The great thing about LLama 2 is that it uses Rotary Positional Embeddings (RoPE) as opposed to the typical sin function encoding. Each Attention layer is modified using that embedding and it ensures the computed attention between input tokens to be only dependent on the distance between those tokens. If token T1 is at position i and a token T2 at position j, the attention A(T1, T2) = f(j - i) is a function of j - i. The attention is not dependent on the specific token's locations but on their relative positions.

The technique they use at Meta to extend the context window is to interpolate at non-integer positions. Basically, if the original window size is L, you can extend it to L' (with L' > L) by rescaling the integer positions

i' = i \* L / L'

As an example, if you wanted to have a text input of 16,384 tokens (so 4x the window size of LLama 2) into LLama 2, you would just need to divide every integer position by 4: i' = i / 4. To be clear, if you look at the implementation of LLama 2 available on GitHub (line 101 in model.py today https://lnkd.in/eyQER_aq), you would just need to replace the following line of code

t = torch.arange(end, device=freqs.device)
by
t = torch.arange(end, device=freqs.device) / 4

How simple is that? Because the model was not trained for that position embedding, you would need to fine-tune the model a bit to adapt it to that new context window and position embedding. When we think that LLama 2 will most likely be used to be fine-tuned on private data, that is the icing on the cake to be able to dynamically adapt the context window to our needs as we fine-tune it.

You can look at the method here: https://lnkd.in/gPUzdBPi. They were able to extend LLama's context window by 16 times while keeping the performance at the same level!

--
👉 Don't forget to subscribe to my ML newsletter: https://lnkd.in/g4iKyRmS

# Chain of thought

<div align="center">
  <img src="https://media.licdn.com/dms/image/D4D22AQFfjm57DwkE0Q/feedshare-shrink_800/0/1711556818856?e=1721865600&v=beta&t=lP3cm8k4vs88DlNm-YEuedgaWD2Y9IDC2uEmzrNkmvk" alt="gd" width="700" height="500" />

</div>
