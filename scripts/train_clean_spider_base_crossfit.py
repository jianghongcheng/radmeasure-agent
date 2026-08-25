#!/usr/bin/env python3
"""Train a data-clean T5-base and produce database-cross-fitted Spider SQL."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party/spider_data/spider_data"


def fold_of(db_id: str, folds: int, seed: int) -> int:
    return int(hashlib.sha256(f"clean-base:{seed}:{db_id}".encode()).hexdigest()[:8], 16) % folds


def schema_strings(tables: list[dict], schema_format="compact", sample_rows=0, allowed_databases=None) -> dict[str, str]:
    output = {}
    for row in tables:
        if allowed_databases is not None and row["db_id"] not in allowed_databases:
            continue
        if schema_format == "create_samples":
            db_path = DATA / "database" / row["db_id"] / f"{row['db_id']}.sqlite"
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            definitions = {};terms={};neighbors=collections.defaultdict(set)
            for table_name, ddl in connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                block = ddl or f"CREATE TABLE {table_name}"
                if sample_rows:
                    quoted = '"' + table_name.replace('"', '""') + '"'
                    cursor = connection.execute(f"SELECT * FROM {quoted} LIMIT {int(sample_rows)}")
                    names = [value[0] for value in cursor.description or []]
                    values = cursor.fetchall()
                    rendered = [[str(cell).replace("\n", " ")[:48] if cell is not None else "NULL" for cell in value] for value in values]
                    block += f"\nSAMPLE {table_name} columns={names} rows={rendered}"
                definitions[table_name]=block;terms[table_name]=set(re.findall(r'[a-z0-9]+',(' '.join([table_name]+names)).lower().replace('_',' ')))
            original_columns=row['column_names_original'];table_names=row['table_names_original']
            for left,right in row['foreign_keys']:
                lt,_=original_columns[left];rt,_=original_columns[right]
                if lt>=0 and rt>=0:neighbors[table_names[lt]].add(table_names[rt]);neighbors[table_names[rt]].add(table_names[lt])
            connection.close();output[row["db_id"]]={'blocks':definitions,'terms':{key:sorted(value) for key,value in terms.items()},'neighbors':{key:sorted(value) for key,value in neighbors.items()}};continue
        table_names = row["table_names_original"]
        columns: dict[int, list[str]] = {index: [] for index in range(len(table_names))}
        for table_index, column in row["column_names_original"]:
            if table_index >= 0:
                columns[table_index].append(column)
        pieces = [f"{table}({', '.join(columns[index])})" for index, table in enumerate(table_names)]
        foreign = []
        original_columns = row["column_names_original"]
        for left, right in row["foreign_keys"]:
            lt, lc = original_columns[left]
            rt, rc = original_columns[right]
            foreign.append(f"{table_names[lt]}.{lc}={table_names[rt]}.{rc}")
        output[row["db_id"]] = " ; ".join(pieces) + (" ; foreign keys: " + ", ".join(foreign) if foreign else "")
    return output


def prompt(row: dict, schemas: dict[str, str], max_schema_tables=8) -> str:
    schema=schemas[row['db_id']]
    if isinstance(schema,dict):
        question=set(re.findall(r'[a-z0-9]+',row['question'].lower().replace('_',' ')));scores=[]
        for table,terms in schema['terms'].items():
            terms=set(terms);overlap=len(question&terms);phrase=sum(term in row['question'].lower() for term in terms if len(term)>2);scores.append((overlap*10+phrase,table))
        ranked=[table for _,table in sorted(scores,key=lambda value:(-value[0],value[1]))];selected=ranked[:max_schema_tables];expanded=list(selected)
        for table in selected:
            for neighbor in schema['neighbors'].get(table,[]):
                if neighbor not in expanded and len(expanded)<max_schema_tables+2:expanded.append(neighbor)
        schema_text='\n\n'.join(schema['blocks'][table] for table in expanded)
    else:schema_text=schema
    return f"translate English to SQL: {row['question']} | schema: {schema_text}"


class SQLDataset(Dataset):
    def __init__(self, rows, schemas, tokenizer, max_input, max_output, labels=True, max_schema_tables=8):
        self.rows, self.schemas, self.tokenizer = rows, schemas, tokenizer
        self.max_input, self.max_output, self.labels, self.max_schema_tables = max_input, max_output, labels, max_schema_tables

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(prompt(row, self.schemas, self.max_schema_tables), max_length=self.max_input, truncation=True)
        item = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "index": index}
        if self.labels:
            item["labels"] = self.tokenizer(row["query"], max_length=self.max_output, truncation=True)["input_ids"]
        return item


def collate(tokenizer, labels=True):
    def function(items):
        indexes = [item.pop("index") for item in items]
        if labels:
            label_values = [item.pop("labels") for item in items]
        batch = tokenizer.pad(items, padding=True, return_tensors="pt")
        if labels:
            width = max(map(len, label_values))
            batch["labels"] = torch.tensor([value + [-100] * (width - len(value)) for value in label_values])
        batch["indexes"] = indexes
        return batch
    return function


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-base")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--max-input", type=int, default=768)
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/research/spider_clean_base")
    parser.add_argument("--schema-format", choices=["compact", "create_samples"], default="compact")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--max-schema-tables", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed + args.fold)
    seal = json.loads((ROOT / "outputs/research/spider_protocol_seal_v1.json").read_text())
    development = set(seal["development_databases"])
    confirmatory = set(seal["confirmatory_databases"])
    all_rows = json.loads((DATA / "train_spider.json").read_text()) + json.loads((DATA / "train_others.json").read_text())
    rows = [{**row, "source_index": index} for index, row in enumerate(all_rows) if row["db_id"] in development]
    assert not ({row["db_id"] for row in rows} & confirmatory)
    train_rows = [row for row in rows if fold_of(row["db_id"], args.folds, args.seed) != args.fold]
    test_rows = [row for row in rows if fold_of(row["db_id"], args.folds, args.seed) == args.fold]
    schemas = schema_strings(json.loads((DATA / "tables.json").read_text()), args.schema_format, args.sample_rows, development)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable();model.config.use_cache=False
    train_data = SQLDataset(train_rows, schemas, tokenizer, args.max_input, args.max_output, max_schema_tables=args.max_schema_tables)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate(tokenizer), num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates = args.epochs * ((len(train_loader) + args.gradient_accumulation - 1) // args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, updates // 20), updates)
    model.train(); optimizer.zero_grad(set_to_none=True); step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            batch.pop("indexes")
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else __import__('contextlib').nullcontext()
            with autocast:
                loss = model(**batch).loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at fold={args.fold}, epoch={epoch}, batch={batch_index}")
            loss.backward(); running += float(loss) * args.gradient_accumulation
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
        print(json.dumps({"fold": args.fold, "epoch": epoch + 1, "mean_loss": running / len(train_loader), "updates": step}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"fold{args.fold}_model"
    model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
    test_data = SQLDataset(test_rows, schemas, tokenizer, args.max_input, args.max_output, labels=False, max_schema_tables=args.max_schema_tables)
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate(tokenizer, False), num_workers=2, pin_memory=True)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            indexes = batch.pop("indexes")
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            generated = model.generate(**batch, max_new_tokens=args.max_output, num_beams=1)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend({"source_index": test_rows[index]["source_index"], "db_id": test_rows[index]["db_id"], "question": test_rows[index]["question"], "predicted_sql": sql} for index, sql in zip(indexes, decoded))
    metadata = {
        "model": args.model, "fold": args.fold, "folds": args.folds, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "precision": args.precision, "max_input": args.max_input, "max_output": args.max_output,
        "schema_format": args.schema_format, "sample_rows": args.sample_rows, "max_schema_tables": args.max_schema_tables,
        "gradient_checkpointing": args.gradient_checkpointing,
        "training_databases": sorted({row["db_id"] for row in train_rows}),
        "test_databases": sorted({row["db_id"] for row in test_rows}),
        "confirmatory_gold_used": False, "predictions": predictions,
    }
    (args.output_dir / f"fold{args.fold}_predictions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"fold": args.fold, "train": len(train_rows), "test": len(test_rows), "prediction_file": str(args.output_dir / f'fold{args.fold}_predictions.json')}), flush=True)


if __name__ == "__main__":
    main()
