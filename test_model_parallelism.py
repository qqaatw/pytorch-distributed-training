import argparse
import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

MAX_GPU_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32

# Copied from accelerate.utils.py::_gpu_gather()
def gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return honor_type(tensor, (_gpu_gather(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: _gpu_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
        )
    if tensor.ndim == 0:
        tensor = tensor.clone()[None]
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


class ConcatBert(torch.nn.Module):
    "Just for test, ignore this."

    def __init__(self, bert_1, bert_2, device):
        super().__init__()
        self.second_device = torch.device("cuda:0")

        self.bert_1 = bert_1
        self.bert_1.to(device)

        self.bert_2 = bert_2
        self.bert_2.to(self.second_device)

    def forward(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)
        input_shape = kwargs["input_ids"].size()

        hidden_states = self.bert_1(*args, **kwargs).last_hidden_state
        extended_attention_mask: torch.Tensor = self.bert_2.get_extended_attention_mask(
            kwargs["attention_mask"], input_shape, self.second_device
        )

        hidden_states = hidden_states.to(self.second_device)
        extended_attention_mask = extended_attention_mask.to(self.second_device)

        encoder_outputs = self.bert_2.bert.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.bert_2.bert.pooler(sequence_output)

        pooled_output = self.bert_2.dropout(pooled_output)
        logits = self.bert_2.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.bert_2.num_labels),
                labels.view(-1).to(self.second_device),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class TriBert(torch.nn.Module):
    def __init__(self, bert_1, bert_2, bert_3, first_device, second_device):
        super().__init__()
        self.first_device = first_device
        self.second_device = second_device

        self.bert_1 = bert_1
        self.bert_1.to(self.first_device)
        self.bert_2 = bert_2
        self.bert_2.to(self.second_device)
        self.bert_3 = bert_3
        self.bert_3.to(self.second_device)

    def forward(self, **kwargs):
        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask")
        token_type_ids = kwargs.pop("token_type_ids")
        labels = kwargs.pop("labels", None)

        print("Input shape ", input_ids.size())

        # We use the same embedding, pooler, and classifier among three BERTs.
        word_embed = self.bert_1.get_input_embeddings()
        pooler = self.bert_1.bert.pooler
        classifier = self.bert_1.classifier

        inputs_embeds = word_embed(input_ids)

        hidden_state_1 = self.bert_1(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        ).hidden_states[-1]
        hidden_state_2 = self.bert_2(
            inputs_embeds=inputs_embeds.to(self.second_device),
            attention_mask=attention_mask.to(self.second_device),
            token_type_ids=token_type_ids.to(self.second_device),
            output_hidden_states=True,
        ).hidden_states[-1]
        hidden_state_3 = self.bert_3(
            inputs_embeds=inputs_embeds.to(self.second_device),
            attention_mask=attention_mask.to(self.second_device),
            token_type_ids=token_type_ids.to(self.second_device),
            output_hidden_states=True,
        ).hidden_states[-1]

        hidden_states = torch.stack(
            (
                hidden_state_1,
                hidden_state_2.to(self.first_device),
                hidden_state_3.to(self.first_device),
            ),
            dim=1,
        )

        hidden_states = torch.mean(hidden_states, dim=1)
        pooled_output = pooler(hidden_states)
        pooled_output = self.bert_1.dropout(pooled_output)
        logits = self.bert_1.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.bert_1.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def training_function(rank, world_size, config, args):
    # Setup
    setup(rank, world_size)

    # Print RANK and WORLD_SIZE
    print("LOCAL_RANK: ", rank)
    print("world_size: ", world_size)
    print("MASTER_ADDR:", os.environ["MASTER_ADDR"])
    print("MASTER_PORT:", os.environ["MASTER_PORT"])

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    correct_bias = config["correct_bias"]
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    device = torch.device("cuda:1")
    second_device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    metric = load_metric("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets.rename_column_("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    bert_1 = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )
    bert_2 = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )
    bert_3 = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )
    model = TriBert(bert_1, bert_2, bert_3, device, second_device)
    print(f"[RANK: {rank}] Initialized model.")

    # Instantiate optimizer
    parameters = model.parameters()
    optimizer = AdamW(params=parameters, lr=lr, correct_bias=correct_bias)
    print(f"[RANK: {rank}] Initialized optimizer.")

    # Prepare everything
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=None,  # For model parallelism
        output_device=None,  # For model parallelism,
        find_unused_parameters=True,
    )
    train_sampler = DistributedSampler(tokenized_datasets["train"])
    train_dataloader = DataLoader(
        dataset=tokenized_datasets["train"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )
    eval_sampler = DistributedSampler(tokenized_datasets["validation"])
    eval_dataloader = DataLoader(
        dataset=tokenized_datasets["validation"],
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
    )
    print(f"[RANK: {rank}] All prepared.")

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            print("Step:", step)

            batch.to(device)
            outputs = model(**batch)

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

            if step % gradient_accumulation_steps != 0:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=gather(predictions),
                references=gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        # print only on the main process.
        if rank == 0:
            print(f"epoch {epoch}:", eval_metric)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    # Mixed precision training is currently not supported.
    # parser.add_argument("--fp16", type=bool, default=False, help="If passed, will use FP16 training.")
    args = parser.parse_args()
    config = {
        "lr": 2e-5,
        "num_epochs": 3,
        "correct_bias": True,
        "seed": 42,
        "batch_size": 96,
    }

    world_size = torch.cuda.device_count()

    mp.spawn(
        training_function, args=(world_size, config, args), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
