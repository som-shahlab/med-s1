Supervised fine-tuning (or SFT for short) is a crucial step in RLHF. In TRL we provide an easy-to-use API to create your SFT models and train them with few lines of code on your dataset.

Check out a complete flexible example at trl/scripts/sft.py. Experimental support for Vision Language Models is also included in the example examples/scripts/sft_vlm.py.

Quickstart
If you have a dataset hosted on the 🤗 Hub, you can easily fine-tune your SFT model using SFTTrainer from TRL. Let us assume your dataset is imdb, the text you want to predict is inside the text field of the dataset, and you want to fine-tune the facebook/opt-350m model. The following code-snippet takes care of all the data pre-processing and training for you:

Copied
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_seq_length=512,
    output_dir="/tmp",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
Make sure to pass the correct value for max_seq_length as the default value will be set to min(tokenizer.model_max_length, 1024).

You can also construct a model outside of the trainer and pass it as follows:

Copied
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

training_args = SFTConfig(output_dir="/tmp")

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
The above snippets will use the default training arguments from the SFTConfig class. If you want to modify the defaults pass in your modification to the SFTConfig constructor and pass them to the trainer via the args argument.

Advanced usage
Train on completions only
You can use the DataCollatorForCompletionOnlyLM to train your model on the generated prompts only. Note that this works only in the case when packing=False. To instantiate that collator for instruction data, pass a response template and the tokenizer. Here is an example of how it would work to fine-tune opt-350m on completions only on the CodeAlpaca dataset:

Copied
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="/tmp"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
To instantiate that collator for assistant style conversation data, pass a response template, an instruction template and the tokenizer. Here is an example of how it would work to fine-tune opt-350m on assistant completions only on the Open Assistant Guanaco dataset:

Copied
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    args=SFTConfig(output_dir="/tmp"),
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
Make sure to have a pad_token_id which is different from eos_token_id which can result in the model not properly predicting EOS (End of Sentence) tokens during generation.

Using token_ids directly for response_template
Some tokenizers like Llama 2 (meta-llama/Llama-2-XXb-hf) tokenize sequences differently depending on whether they have context or not. For example:

Copied
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))

prompt = """### User: Hello\n\n### Assistant: Hi, how can I help you?"""
print_tokens_with_ids(prompt)  # [..., ('▁Hello', 15043), ('<0x0A>', 13), ('<0x0A>', 13), ('##', 2277), ('#', 29937), ('▁Ass', 4007), ('istant', 22137), (':', 29901), ...]

response_template = "### Assistant:"
print_tokens_with_ids(response_template)  # [('▁###', 835), ('▁Ass', 4007), ('istant', 22137), (':', 29901)]
In this case, and due to lack of context in response_template, the same string (”### Assistant:”) is tokenized differently:

Text (with context): [2277, 29937, 4007, 22137, 29901]
response_template (without context): [835, 4007, 22137, 29901]
This will lead to an error when the DataCollatorForCompletionOnlyLM does not find the response_template in the dataset example text:

Copied
RuntimeError: Could not find response key [835, 4007, 22137, 29901] in token IDs tensor([    1,   835,  ...])
To solve this, you can tokenize the response_template with the same context as in the dataset, truncate it as needed and pass the token_ids directly to the response_template argument of the DataCollatorForCompletionOnlyLM class. For example:

Copied
response_template_with_context = "\n### Assistant:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
Add Special Tokens for Chat Format
Adding special tokens to a language model is crucial for training chat models. These tokens are added between the different roles in a conversation, such as the user, assistant, and system and help the model recognize the structure and flow of a conversation. This setup is essential for enabling the model to generate coherent and contextually appropriate responses in a chat environment. The setup_chat_format() function in trl easily sets up a model and tokenizer for conversational AI tasks. This function:

Adds special tokens to the tokenizer, e.g. <|im_start|> and <|im_end|>, to indicate the start and end of a conversation.
Resizes the model’s embedding layer to accommodate the new tokens.
Sets the chat_template of the tokenizer, which is used to format the input data into a chat-like format. The default is chatml from OpenAI.
optionally you can pass resize_to_multiple_of to resize the embedding layer to a multiple of the resize_to_multiple_of argument, e.g. 64. If you want to see more formats being supported in the future, please open a GitHub issue on trl
Copied
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Set up the chat format with default 'chatml' format
model, tokenizer = setup_chat_format(model, tokenizer)
With our model and tokenizer set up, we can now fine-tune our model on a conversational dataset. Below is an example of how a dataset can be formatted for fine-tuning.

Dataset format support
The SFTTrainer supports popular dataset formats. This allows you to pass the dataset to the trainer without any pre-processing directly. The following formats are supported:

conversational format
Copied
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "..."}]}
instruction format
Copied
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
If your dataset uses one of the above formats, you can directly pass it to the trainer without pre-processing. The SFTTrainer will then format the dataset for you using the defined format from the model’s tokenizer with the apply_chat_template method.

Copied
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

...

# load jsonl dataset
dataset = load_dataset("json", data_files="path/to/dataset.jsonl", split="train")
# load dataset from the HuggingFace Hub
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

...

training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    args=training_args,
    train_dataset=dataset,
)
If the dataset is not in one of those format you can either preprocess the dataset to match the formatting or pass a formatting function to the SFTTrainer to do it for you. Let’s have a look.

Format your input prompts
For instruction fine-tuning, it is quite common to have two columns inside the dataset: one for the prompt & the other for the response. This allows people to format examples like Stanford-Alpaca did as follows:

Copied
Below is an instruction ...

### Instruction
{prompt}

### Response:
{completion}
Let us assume your dataset has two fields, question and answer. Therefore you can just run:

Copied
...
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

trainer.train()
To properly format your input make sure to process all the examples by looping over them and returning a list of processed text. Check out a full example of how to use SFTTrainer on alpaca dataset here

Packing dataset ( ConstantLengthDataset )
SFTTrainer supports example packing, where multiple short examples are packed in the same input sequence to increase training efficiency. This is done with the ConstantLengthDataset utility class that returns constant length chunks of tokens from a stream of examples. To enable the usage of this dataset class, simply pass packing=True to the SFTConfig constructor.

Copied
...
training_args = SFTConfig(packing=True)

trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args
)

trainer.train()
Note that if you use a packed dataset and if you pass max_steps in the training arguments you will probably train your models for more than few epochs, depending on the way you have configured the packed dataset and the training protocol. Double check that you know and understand what you are doing. If you don’t want to pack your eval_dataset, you can pass eval_packing=False to the SFTConfig init method.

Customize your prompts using packed dataset
If your dataset has several fields that you want to combine, for example if the dataset has question and answer fields and you want to combine them, you can pass a formatting function to the trainer that will take care of that. For example:

Copied
def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text

training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func
)

trainer.train()
You can also customize the ConstantLengthDataset much more by directly passing the arguments to the SFTConfig constructor. Please refer to that class’ signature for more information.