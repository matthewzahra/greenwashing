import pandas as pd
import numpy as np
from datasets import (
    load_dataset,
    concatenate_datasets
)
from sklearn.model_selection import StratifiedKFold

def prepare_dataset(ds_name):
    # df = pd.read_csv(ds_name, encoding="utf-8", skiprows=1)
    # df["context"] = np.where(pd.isna(df["Analyst Context"]), df["Context"], df["Analyst Context"])
    # df["Result (Does the Proofpoint apply?)"] = df["Result (Does the Proofpoint apply?)"].apply(lambda x: 1 if x=="yes" else 0)
    # df.rename(columns={"Result (Does the Proofpoint apply?)": "labels", "Proofpoint Match Words": "matchwords"}, inplace=True)
    # df = df[["context", "matchwords", "labels"]]
    # df = df.dropna()
    # return df

    dataset = load_dataset('csv', data_files=ds_name)
    # drop empty entries
    dataset = dataset.filter(lambda example: example['context'] != '' or example['proofpoint'] != '' or example['labels'] != '')
    return dataset["train"]

def load_train_test_dataset(ds_name, train_size=0.8, random_state=200, test_run=False):
    ds = prepare_dataset(ds_name)
    if test_run:
        ds = ds.select(range(100))
    if (train_size == 1):
        train_dataset = ds
        eval_dataset = []
    else:
        train_validation_split = ds.train_test_split(train_size=train_size, seed=random_state)
        train_dataset = train_validation_split['train']
        eval_dataset = train_validation_split['test']
    # if not test_run:
    #     balance_dataset(train_dataset, "labels")
    return train_dataset, eval_dataset

def load_cross_validation_dataset(train_ds, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=200)
    return skf.split(np.zeros(len(train_ds)), train_ds["labels"])

import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
from load_dataset import (
    load_train_test_dataset,
    load_cross_validation_dataset
)
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score
import wandb

test_run_train_size = 20
test_run_eval_size = 20

class AccuracyCallback(TrainerCallback):
    def __init__(self) -> None:
        self.last_accuracy = None

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if self.last_accuracy is not None and metrics['eval_accuracy'] < self.last_accuracy:
            control.should_stop = True
        self.last_accuracy = metrics['eval_accuracy']

def check_memory_alloc_reserved():
    print("Memory Reserved: ", torch.cuda.memory_reserved())
    print("Memory Allocated: ", torch.cuda.memory_allocated())

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted',zero_division=0)
    recall = recall_score(labels, preds, average='weighted',zero_division=0)
    f1 = 2/((1/precision)+(1/recall))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_trainer(args, model, tokenizer, train_dataset, eval_dataset):
    logging_steps = 20 if args.test else ((len(train_dataset)*args.num_epochs) // args.batch_size)//25
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # output directory
        num_train_epochs=args.num_epochs,    # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        logging_steps=logging_steps,          # log every x updates steps
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,                       
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,            # evaluation dataset
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    return trainer

def get_qlora_model(args):

    # check if gpu is available
    print("GPU Available: ", torch.cuda.is_available())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, quantization_config=bnb_config, device_map='auto')
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def load_model_tokenizer(args):
    model = get_qlora_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def tokenize_preprocess_dataset(train_ds, test_ds, tokenizer):
    def preprocess_function(examples):
        text = ["Proofpoint "+p+". Context: "+c+" Answer:" for (p, c) in zip(examples["proofpoint"], examples["context"])]
        return {"text": text, "label": examples["labels"]}

    def tokenizer_func(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=["proofpoint", "context"])
    test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=["proofpoint", "context"])
    
    train_ds = train_ds.map(tokenizer_func, batched=True)
    test_ds = test_ds.map(tokenizer_func, batched=True)

    return train_ds, test_ds

def main_train_loop(args, train_ds, test_ds):
    model, tokenizer = load_model_tokenizer(args)
    train_ds, test_ds = tokenize_preprocess_dataset(train_ds, test_ds, tokenizer)

    wandb.init(project=args.wandb_project, name=args.run_name, entity="GDP-Ample")
    wandb.config.update(args)

    trainer = get_trainer(args, model, tokenizer, train_ds, test_ds)
    accuracy_callback = AccuracyCallback()
    trainer.add_callback(accuracy_callback)
    trainer.train()

    wandb.finish()

    trainer.save_model(args.output_dir)
    return accuracy_callback.last_accuracy

def train_model(args):
    ds_name = "cleanedFinal.csv"
    train_size = 1 if args.cross_validation else 0.8
    train_dataset, eval_dataset = load_train_test_dataset(ds_name, train_size=train_size, test_run=args.test)

    if args.cross_validation:
        train_test_splits = load_cross_validation_dataset(train_dataset, args.n_splits)

    check_memory_alloc_reserved()

    average_accuracy = 0
    run_name = args.run_name
    output_dir = args.output_dir

    if args.cross_validation:
        for i, (train_ind, test_ind) in enumerate(train_test_splits):
            curr_train_ds = train_dataset.select(train_ind)
            curr_test_ds = train_dataset.select(test_ind)
            args.output_dir = output_dir+f"/fold_{i}"
            args.run_name = run_name+f"-fold-{i}"
            accuracy = main_train_loop(args, curr_train_ds, curr_test_ds)
            average_accuracy += accuracy
    else:
        main_train_loop(args, train_dataset, eval_dataset)

    if args.cross_validation:
        average_accuracy /= args.n_splits
        wandb.init(project=args.wandb_project, name=run_name+'final', entity="GDP-Ample")
        wandb.config.update(args)
        wandb.log({"Average Accuracy": average_accuracy})
        wandb.finish()


class ArgumentForLLM:
    def __init__(self) -> None:
        self.model_name = "EleutherAI/pythia-1b"
        self.output_dir = "Checkpoints"
        self.batch_size = 8
        self.num_epochs = 10
        self.learning_rate = 5e-5
        self.test = False
        self.run_name = "run1"
        self.wandb_project = "Testing-Pythia-1b"
        self.cross_validation = True
        self.n_splits = 5   

args = ArgumentForLLM()
train_model(args)