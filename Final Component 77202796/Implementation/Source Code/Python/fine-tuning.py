import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, load_metric
import re
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer

import torchaudio
import random
import gc

import librosa
import numpy as np
from sys import getsizeof

from DataCollatorCTCWithPadding import DataCollatorCTCWithPadding

prefix_path = "./content/ne_np_female/"
filename = 'line_index.tsv'
colnames = ['fpath', 'text']
df = pd.read_csv(prefix_path + filename, sep='\t', header=None, names=colnames)
# Folder structure for the files
# df['path'] = prefix_path + 'data/' + df['fpath'].str[:2]
# # Add file link to the path
# df['path'] = df['path'] + "/" + df['fpath'] + '.flac'

df['path'] = prefix_path + 'wavs/' + df['fpath'] + '.wav'


# Drop Duplicates
df = df.drop_duplicates(subset=['text'])

# Test/Train split 10%
x_train, x_test = train_test_split(df, train_size=0.1)
# Load Dataset from dataframe
dataset_test = Dataset.from_pandas(x_test)
dataset_train = Dataset.from_pandas(x_train)


dataset_test = dataset_test.remove_columns(['fpath',  '__index_level_0__'])
dataset_train = dataset_train.remove_columns(['fpath', '__index_level_0__'])

# Remove characters to be ignored using regex
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


# Function to remove special characters/characters to ignore
def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex, '', batch['text']).lower() + " "


dataset_test = dataset_test.map(remove_special_characters)
dataset_train = dataset_train.map(remove_special_characters)


# Function to extract all the individual characters in the transcripts
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# Mapping the datasets to the extract_all_chars function and generate the vocabulary
vocab_test = dataset_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                              remove_columns=dataset_test.column_names)
vocab_train = dataset_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                remove_columns=dataset_train.column_names)
# Generate the complete vocabulary from both test and train vocabularies
vocab_list = list(set(vocab_train['vocab'][0]) | set(vocab_test['vocab'][0]))

# Generate the vocabulary dictionary
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Remove unwanted characters from the vocabulary
try:
    del vocab_dict['\\']
    del vocab_dict['a']
    del vocab_dict['b']
    del vocab_dict['c']
    del vocab_dict['e']
    del vocab_dict['f']
    del vocab_dict['k']
    del vocab_dict['o']
    del vocab_dict['\xa0']
    del vocab_dict['\u200c']
    del vocab_dict['\u200d']
    del vocab_dict['\u200e']
    del vocab_dict['\u200f']
except:
    pass

# Replace the space " " key to a visible character "|"
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
# Create keys for unknown and paddings
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab_finetune.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Create a Wav2Vec2CTCTokenizer object with the generated vocabulary json with the required tokens
tokenizer = Wav2Vec2CTCTokenizer("./vocab_finetune.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Create a Wav2Vec2FeatureExtractor object for the audio
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

# Create Wav2Vec2Processor object using the created feature_extractor and tokenizer
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

sum = 0;


# Convert audio to numpy array for processing
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    if random.random() < 0.25:
        gc.collect()
    # print(gc.garbage)
    # global sum;
    # sum = sum + getsizeof(batch["speech"])
    # print(sum)
    return batch


dataset_test = dataset_test.map(speech_file_to_array_fn, remove_columns=dataset_test.column_names)
dataset_train = dataset_train.map(speech_file_to_array_fn, remove_columns=dataset_train.column_names)


# Resample the audio to 16000Hz
def resample(batch):
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), 48_000, 16_000)
    batch["sampling_rate"] = 16_000
    return batch


# Map and resample all the audio
dataset_train = dataset_train.map(resample, num_proc=4)
dataset_test = dataset_test.map(resample, num_proc=4)


def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


dataset_train = dataset_train.map(prepare_dataset, remove_columns=dataset_train.column_names, batch_size=8, num_proc=4,
                                  batched=True)
dataset_test = dataset_test.map(prepare_dataset, remove_columns=dataset_test.column_names, batch_size=8, num_proc=4,
                                batched=True)
# Data collater object
# Add blanks i.e. padding to the beginning to make the length of all the audio data array same
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Metric for WER calculation
wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "./wav2vec2-xlsr-nepali",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

# Trainer Arguments
training_args = TrainingArguments(
    output_dir="/content/wav2vec2-large-xlsr-nepali",
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=5,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=1,
)
# Create model trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    tokenizer=processor.feature_extractor,
)
# Start training
trainer.train()
