import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        """
        T5 text-to-SQL dataset.

        data_folder: normally 'data_preprocessed'
        split: 'train', 'dev', or 'test'
        """
        self.split = split
        self.data_folder = data_folder
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        self.encoder_ids = []
        self.decoder_inputs = []
        self.decoder_targets = []
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
            assert len(sql_lines) == len(nl_lines)

        # BOS token: use extra_id_0
        bos_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

        for i, nl in enumerate(nl_lines):
            enc = tokenizer(
                nl,
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

                # teacher forcing: shift right
                dec_in_ids = torch.full(
                    (tgt_ids.size(0),), PAD_IDX, dtype=torch.long
                )
                dec_in_ids[0] = bos_id
                if tgt_ids.size(0) > 1:
                    dec_in_ids[1:] = tgt_ids[:-1]

                self.decoder_targets.append(tgt_ids)
                self.decoder_inputs.append(dec_in_ids)
                self.initial_decoder_inputs.append(
                    torch.tensor(bos_id, dtype=torch.long)
                )
            else:
                # test: only encoder + BOS for generation
                self.initial_decoder_inputs.append(
                    torch.tensor(bos_id, dtype=torch.long)
                )

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split != "test":
            return (
                self.encoder_ids[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx],
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
      batch elems: (encoder_ids, decoder_inputs, decoder_targets, initial_decoder_input)
    """
    enc_ids, dec_in, dec_tgt, init_dec = zip(*batch)

    enc_padded = pad_sequence(enc_ids, batch_first=True, padding_value=PAD_IDX)
    dec_in_padded = pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX)
    dec_tgt_padded = pad_sequence(dec_tgt, batch_first=True, padding_value=PAD_IDX)

    encoder_mask = (enc_padded != PAD_IDX).long()
    init_dec = torch.stack(init_dec)  # (B,)

    return enc_padded, encoder_mask, dec_in_padded, dec_tgt_padded, init_dec

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

def get_dataloader(batch_size, split):
    # *** IMPORTANT: train on preprocessed data ***
    data_folder = 'data_preprocessed'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader

# def load_lines(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#     return lines

# def load_prompting_data(data_folder):
#     # TODO
#     return train_x, train_y, dev_x, dev_y, test_x