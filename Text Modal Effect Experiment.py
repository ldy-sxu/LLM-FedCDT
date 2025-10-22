import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertModel, AutoModelForSequenceClassification, BertTokenizer
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate
from typing import cast, OrderedDict, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import random
import jieba
import logging

jieba.setLogLevel(logging.CRITICAL)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NUMEXPR_MAX_THREADS'] = '20'


class AttentiveFusion(nn.Module):
    def __init__(self, img_dim: int, txt_dim: int):
        super().__init__()

        hidden_dim = 128

        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        self.attention_vector = nn.Linear(hidden_dim, 1)

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor):
        h_img = torch.tanh(self.img_proj(x_img))
        h_txt = torch.tanh(self.txt_proj(x_txt))

        x_concat = torch.stack([h_img, h_txt], dim=1)

        score = self.attention_vector(x_concat)

        weights = torch.softmax(score.squeeze(-1), dim=1)

        w_img = weights[:, 0].unsqueeze(1)
        w_txt = weights[:, 1].unsqueeze(1)

        gated_x_img = x_img * w_img
        gated_x_txt = x_txt * w_txt

        fused_output = torch.cat([gated_x_img, gated_x_txt], dim=1)

        return fused_output, weights


class MultiModalNet(nn.Module):
    def __init__(self, num_classes: int, bert_name: str = "bert-base-chinese",
                 use_image: bool = True, use_text: bool = True, device: torch.device = torch.device('cpu')):
        super().__init__()

        self.use_image = use_image
        self.use_text = use_text
        self.device = device

        if not self.use_image and not self.use_text:
            raise ValueError("At least one of the image or text modalities must be enabled.")

        self.img_encoder = None
        self.img_feat_dim = 0
        if self.use_image:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.img_encoder = nn.Sequential(
                *list(resnet.children())[:-1],
            )
            self.img_feat_dim = resnet.fc.in_features

        self.txt_encoder = None
        self.txt_feat_dim = 0
        if self.use_text:
            pre_trained_bert_path = bert_name

            if not os.path.exists(pre_trained_bert_path):
                print(f"Warning: Pretrained BERT model not found in path '{pre_trained_bert_path}'.")
                temp_bert_model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese',
                                                                                     num_labels=num_classes)
            else:
                temp_bert_model = AutoModelForSequenceClassification.from_pretrained(pre_trained_bert_path)

            self.txt_encoder = temp_bert_model.bert

            for param in self.txt_encoder.parameters():
                param.requires_grad = False

            self.txt_encoder.to(self.device)

            self.txt_feat_dim = self.txt_encoder.config.hidden_size

        fused_output_dim = 0
        if self.use_image and self.use_text:
            self.fusion_module = AttentiveFusion(self.img_feat_dim, self.txt_feat_dim)
            fused_output_dim = self.img_feat_dim + self.txt_feat_dim
        elif self.use_image:
            fused_output_dim = self.img_feat_dim
        elif self.use_text:
            fused_output_dim = self.txt_feat_dim
        else:
            raise ValueError("No modality encoder is enabled.")

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, img, input_ids, attention_mask):
        x_img = None
        x_txt = None

        if self.use_image:
            x_img = self.img_encoder(img)
            x_img = x_img.view(x_img.size(0), -1)

        if self.use_text:
            self.txt_encoder.eval()
            with torch.no_grad():
                out = self.txt_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            x_txt = out.last_hidden_state[:, 0, :]

        if self.use_image and self.use_text:
            x, _ = self.fusion_module(x_img, x_txt)
        elif self.use_image:
            x = x_img
        elif self.use_text:
            x = x_txt
        else:
            raise ValueError("No valid modality input.")

        fused_features = self.fusion_mlp(x)
        logits = self.classifier(fused_features)
        return logits


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
                 transform=None,
                 text_modification_type: str = "original",
                 mask_ratio: float = 0.0):
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
            df["编号"].astype(str),
            df["医学语义描述"].astype(str)
        ))

        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.text_modification_type = text_modification_type
        self.mask_ratio = mask_ratio

    def _mask_text(self, text: str) -> str:
        """Mask part of the text according to the specified ratio"""
        words = list(jieba.cut(text, cut_all=False))
        num_words_to_mask = int(len(words) * self.mask_ratio)
        if num_words_to_mask > 0:
            indices_to_mask = random.sample(range(len(words)), num_words_to_mask)
            masked_words = [word for i, word in enumerate(words) if i not in indices_to_mask]
            return "".join(masked_words)
        return text

    def _permute_text(self, text: str) -> str:
        """Randomly permute the words in the text"""
        words = list(jieba.cut(text, cut_all=False))
        random.shuffle(words)
        return "".join(words)

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

        if self.text_modification_type == "mask":
            text = self._mask_text(text)
        elif self.text_modification_type == "permute":
            text = self._permute_text(text)

        tok = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        return img, input_ids, attention_mask, label


class CentralizedDataProcessor:
    def __init__(self, img_dir: str, label_txt: str, desc_xlsx: str,
                 batch_size: int,
                 tokenizer_name: str = "bert-base-chinese", max_length: int = 256,
                 transform=None, test_ratio: float = 0.2,
                 text_modification_type: str = "original",
                 mask_ratio: float = 0.0):

        full = CDTMultiModalDataset(
            img_dir, label_txt, desc_xlsx, tokenizer_name, max_length, transform,
            text_modification_type, mask_ratio
        )
        self.dataset = full
        total_size = len(full)

        all_indices = np.random.permutation(total_size)

        test_size = max(1, int(total_size * test_ratio))
        test_size = min(test_size, total_size - 1)

        self.train_indices = all_indices[:-test_size].tolist()
        self.test_indices = all_indices[-test_size:].tolist()

        self.batch_size = batch_size
        self.test_ratio = test_ratio

    def get_train_dataloader(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        train_sampler = SubsetRandomSampler(self.train_indices)

        return DataLoader(
            self.dataset, batch_size=bs, sampler=train_sampler, num_workers=4,
            pin_memory=True, collate_fn=custom_collate_fn, drop_last=True
        )

    def get_test_dataloader(self, batch_size: int = None):
        bs = batch_size or self.batch_size
        test_sampler = SubsetRandomSampler(self.test_indices)

        return DataLoader(
            self.dataset, batch_size=bs, sampler=test_sampler, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
        )


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, device: torch.device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        img, input_ids, attention_mask, label = batch

        if model.use_image:
            img = img.to(device)
        if model.use_text:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        label = label.to(device)

        logits = model(img, input_ids, attention_mask)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def evaluate_metrics(model, dataloader, loss_fn, device, num_classes):
    """Evaluate model performance and return metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, input_ids, attention_mask, label in dataloader:
            if model.use_image: img = img.to(device)
            if model.use_text:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            label = label.to(device)

            logits = model(img, input_ids, attention_mask)
            loss = loss_fn(logits, label)
            total_loss += loss.item() * label.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cohen_kappa": kappa
    }


def parse_args():
    p = argparse.ArgumentParser("Centralized MultiModal CDT")
    p.add_argument("--img_dir", type=str,
                   default=r"...",
                   help="Image data directory")
    p.add_argument("--label_txt", type=str,
                   default=r"...",
                   help="Label file path")
    p.add_argument("--desc_xlsx", type=str,
                   default=r"...",
                   help="Text description file path")
    p.add_argument("--bert_model", type=str,
                   default=r"...",
                   help="BERT model name")

    p.add_argument("--total_epochs", type=int, default=50, help="Total training epochs for centralized learning")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--max_len", type=int, default=128, help="BERT maximum text length")
    p.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    p.add_argument("--output_dir", type=str, default="./output_centralized", help="Result output directory")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_dir, "centralized_metrics_robustness.csv")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}. Did you forget to use --gpu?")
    print("-" * 30)

    print("\nPlease select training mode:")
    print("  [1] Use only ResNet-18 (image)")
    print("  [2] Use only BERT (text)")
    print("  [3] Use ResNet-18 + BERT (multimodal)")
    choice_modality = input("Please enter 1, 2, or 3 and press Enter: ").strip()

    use_image = False
    use_text = False

    if choice_modality == "1":
        use_image = True
        print("==> Selected mode: Image only")
    elif choice_modality == "2":
        use_text = True
        print("==> Selected mode: Text only")
    elif choice_modality == "3":
        use_image = True
        use_text = True
        print("==> Selected mode: Multimodal")
    else:
        print("Invalid selection, defaulting to multimodal mode.")
        use_image = True
        use_text = True
    print("-" * 30)

    text_modification_type = "original"
    mask_ratio = 0.0

    if use_text:
        print("\nPlease select text robustness experiment mode:")
        print("  [1] Original text (baseline comparison)")
        print("  [2] Random masking (simulate text information loss)")
        print("  [3] Random permutation (simulate text structure destruction)")
        choice_text = input("Please enter 1, 2, or 3 and press Enter: ").strip()

        if choice_text == "2":
            text_modification_type = "mask"
            ratio_input = input("Please enter text masking ratio (e.g., 0.2, 0.5, 0.8): ").strip()
            try:
                mask_ratio = float(ratio_input)
                print(f"==> Selected mode: Random masking with ratio {mask_ratio}")
            except ValueError:
                print("Invalid input, defaulting to original text mode.")
                text_modification_type = "original"
        elif choice_text == "3":
            text_modification_type = "permute"
            print("==> Selected mode: Random text permutation training")
        else:
            print("==> Selected mode: Original text")
    else:
        print("Text modality not enabled, skipping text robustness settings.")

    print("-" * 30)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_processor = CentralizedDataProcessor(
        img_dir=args.img_dir, label_txt=args.label_txt, desc_xlsx=args.desc_xlsx,
        batch_size=args.batch_size,
        tokenizer_name=args.bert_model, max_length=args.max_len, transform=transform,
        text_modification_type=text_modification_type,
        mask_ratio=mask_ratio
    )

    train_loader = data_processor.get_train_dataloader(args.batch_size)
    test_loader = data_processor.get_test_dataloader(args.batch_size)
    print(f"Training set size: {len(train_loader.sampler)} batches: {len(train_loader)}")
    print(f"Test set size: {len(test_loader.sampler)} batches: {len(test_loader)}")
    print("-" * 30)

    num_classes = 5
    model = MultiModalNet(
        num_classes,
        bert_name=args.bert_model,
        use_image=use_image,
        use_text=use_text,
        device=device
    ).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    metrics_history = []

    print(">>> Starting centralized training <<<")

    for epoch in range(1, args.total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        metrics = evaluate_metrics(model, test_loader, criterion, device, num_classes)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics_history.append(metrics)

        print(f"=== Epoch {epoch}/{args.total_epochs} ===")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Test Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-score (weighted): {metrics['f1_score']:.4f}")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print("-" * 30)

    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\nEvaluation metrics for all epochs have been saved to: {metrics_csv_path}")

    if not df_metrics.empty:
        accuracies = df_metrics["accuracy"].values
        max_accuracy = accuracies.max()
        convergence_threshold = 0.95 * max_accuracy

        convergence_epoch = -1
        for i, acc in enumerate(accuracies):
            if acc >= convergence_threshold:
                convergence_epoch = i + 1
                break

        print(
            f"Model convergence epoch (Accuracy reaches 95% of maximum value): {convergence_epoch if convergence_epoch != -1 else 'Not achieved'}")


if __name__ == "__main__":
    main()
