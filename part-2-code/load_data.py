import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
SCHEMA_PATH = os.path.join("data", "flight_database.schema")


class T5Dataset(Dataset):
    def __init__(self, data_folder, split, use_schema: bool = False):
        """
        T5 text-to-SQL dataset.

        data_folder: 'data' or 'data_preprocessed'
        split: 'train', 'dev', or 'test'
        use_schema: if True, prepend the DB schema to each NL query
        """
        self.split = split
        self.data_folder = data_folder
        self.use_schema = use_schema
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # Load schema text once if requested
        self.schema_text = None
        if self.use_schema:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                # Simple flattening: join non-empty lines with spaces
                lines = [line.strip() for line in f if line.strip()]
                # Take first 50 lines to avoid making inputs too long
                self.schema_text = " ".join(lines[:50])
                print(f"Schema loaded for {split}: {len(self.schema_text)} characters")

        self.encoder_ids = []
        self.decoder_targets = []  # No more decoder_inputs!
        self.initial_decoder_inputs = []

        self.process_data(data_folder, split, self.tokenizer)
        
    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        with open(nl_path, "r", encoding="utf-8") as f:
            nl_lines = [l.strip() for l in f if l.strip()]

        sql_lines = None
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            with open(sql_path, "r", encoding="utf-8") as f:
                sql_lines = [l.strip() for l in f if l.strip()]
            assert len(sql_lines) == len(nl_lines), f"Mismatch: {len(nl_lines)} NL vs {len(sql_lines)} SQL"

        # BOS token for generation only
        bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

        for i, nl in enumerate(nl_lines):
            # Optionally add schema context
            if self.schema_text is not None:
                encoder_text = f"translate to SQL: schema: {self.schema_text} query: {nl}"
            else:
                encoder_text = f"translate to SQL: {nl}"

            enc = tokenizer(
                encoder_text,
                truncation=True,
                max_length=512,
                padding=False,
                return_attention_mask=False,
            )
            enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            self.encoder_ids.append(enc_ids)

            if split != "test":
                sql = sql_lines[i]
                dec = tokenizer(
                    sql,
                    truncation=True,
                    max_length=256,
                    padding=False,
                    return_attention_mask=False,
                )
                tgt_ids = torch.tensor(dec["input_ids"], dtype=torch.long)

                # CRITICAL: Store targets WITHOUT shifting!
                # T5's forward() will handle the shifting when labels are passed
                self.decoder_targets.append(tgt_ids)
                self.initial_decoder_inputs.append(torch.tensor(bos_id, dtype=torch.long))
            else:
                # test: only BOS for generation
                self.initial_decoder_inputs.append(torch.tensor(bos_id, dtype=torch.long))

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split != "test":
            return (
                self.encoder_ids[idx],
                self.decoder_targets[idx],  # No decoder_inputs!
                self.initial_decoder_inputs[idx],
            )
        else:
            return (
                self.encoder_ids[idx],
                self.initial_decoder_inputs[idx],
            )


def normal_collate_fn(batch):
    """
    For train/dev:
      batch elems: (encoder_ids, decoder_targets, initial_decoder_input)
    """
    enc_ids, dec_tgt, init_dec = zip(*batch)

    enc_padded = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    dec_tgt_padded = pad_sequence(dec_tgt, batch_first=True, padding_value=PAD_IDX)

    encoder_mask = (enc_padded != PAD_IDX).long()
    init_dec = torch.stack(init_dec)  # (B,)

    return enc_padded, encoder_mask, dec_tgt_padded, init_dec


def test_collate_fn(batch):
    """
    For test:
      batch elems: (encoder_ids, initial_decoder_input)
    """
    enc_ids, init_dec = zip(*batch)

    enc_padded = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (enc_padded != PAD_IDX).long()
    init_dec = torch.stack(init_dec)  # (B,)

    return enc_padded, encoder_mask, init_dec


def get_dataloader(batch_size, split, use_schema: bool = False, use_preprocessed: bool = True):
    """
    Get dataloader for a specific split.
    
    Args:
        batch_size: Batch size
        split: 'train', 'dev', or 'test'
        use_schema: Whether to include schema in encoder input
        use_preprocessed: Whether to use preprocessed data (default: True)
    """
    # Choose data folder based on use_preprocessed flag
    data_folder = "data_preprocessed" if use_preprocessed else "data"
    print(f"Loading {split} data from: {data_folder} (schema={use_schema})")
    
    dset = T5Dataset(data_folder, split, use_schema=use_schema)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size, use_schema=False, use_preprocessed=True):
    """
    Load train, dev, and test dataloaders.
    
    Args:
        batch_size: Training batch size
        test_batch_size: Eval/test batch size
        use_schema: Whether to include schema
        use_preprocessed: Whether to use preprocessed data
    """
    train_loader = get_dataloader(batch_size, "train", use_schema=use_schema, use_preprocessed=use_preprocessed)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema=use_schema, use_preprocessed=use_preprocessed)
    test_loader = get_dataloader(test_batch_size, "test", use_schema=use_schema, use_preprocessed=use_preprocessed)

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines