import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate
from transformers import BertTokenizer
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
import os


def dirichlet_split(labels, num_clients, alpha):
    labels = np.array(labels)
    labels = np.clip(labels, 0, 4)
    class_idx = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    client_indices = defaultdict(list)
    for c, idx in class_idx.items():
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(idx)).astype(int).tolist()
        counts[-1] = len(idx) - sum(counts[:-1])
        start = 0
        for client_id, cnt in enumerate(counts):
            end = start + cnt
            client_indices[client_id] += idx[start:end].tolist()
            start = end
    return client_indices


def custom_collate_fn(batch):
    imgs, input_ids_list, attention_mask_list, labels_list = zip(*batch)
    collated_imgs = default_collate(imgs)
    collated_input_ids = default_collate(input_ids_list)
    collated_attention_mask = default_collate(attention_mask_list)
    collated_labels = torch.tensor(labels_list, dtype=torch.long)
    collated_labels = torch.clamp(collated_labels, 0, 4)
    return collated_imgs, collated_input_ids, collated_attention_mask, collated_labels


class CDTMultiModalDataset(Dataset):
    def __init__(self, img_dir: str, label_txt: str, desc_xlsx: str,
                 tokenizer_name: str = "bert-base-chinese", max_length: int = 256,
                 transform=None):
        self.samples = []
        with open(label_txt, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                idx = parts[0]
                scores = list(map(int, parts[1:5]))
                total = sum(scores)
                label = min(max(total, 0), 4)
                self.samples.append((idx, label))

        df = pd.read_excel(desc_xlsx, engine="openpyxl")
        self.desc_map = dict(zip(
            df["Serial Number"].astype(str),
            df["Medical semantic description"].astype(str)
        ))

        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, label = self.samples[idx]

        img_path = os.path.join(self.img_dir, f"{key}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image {img_path} not found. Using black placeholder.")
            img = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            img = self.transform(img)

        text = self.desc_map.get(key, "")
        tok = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        return img, input_ids, attention_mask, label


class PartitionedMultiModalCDT:
    def __init__(self, img_dir: str, label_txt: str, desc_xlsx: str,
                 num_clients: int, batch_size: int, alpha: float = 0.5,
                 tokenizer_name: str = "bert-base-chinese", max_length: int = 256,
                 transform=None, test_ratio: float = 0.2):
        full = CDTMultiModalDataset(
            img_dir, label_txt, desc_xlsx, tokenizer_name, max_length, transform
        )
        self.dataset = full
        total_size = len(full)

        all_indices = np.random.permutation(total_size)
        test_size = int(total_size * test_ratio)
        self.train_indices = all_indices[:-test_size].tolist()
        self.test_indices = all_indices[-test_size:].tolist()

        train_labels = [full.samples[i][1] for i in self.train_indices]
        train_labels = [int(l) for l in train_labels]

        relative_client_indices = dirichlet_split(train_labels, num_clients, alpha)

        self.indices_dict = {
            client_id: [self.train_indices[relative_idx]
                        for relative_idx in relative_indices]
            for client_id, relative_indices in relative_client_indices.items()
        }

        self.num_clients = num_clients
        self.batch_size = batch_size
        self.alpha = alpha
        self.test_ratio = test_ratio

    def get_dataloader(self, client_id: int, batch_size: int = None):
        bs = batch_size or self.batch_size
        sampler = SubsetRandomSampler(self.indices_dict[client_id])

        return DataLoader(
            self.dataset, batch_size=bs, sampler=sampler, num_workers=4,
            pin_memory=True, collate_fn=custom_collate_fn, drop_last=True
        )

    def get_test_dataloader(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        test_sampler = SubsetRandomSampler(self.test_indices)

        return DataLoader(
            self.dataset, batch_size=bs, sampler=test_sampler, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
        )
