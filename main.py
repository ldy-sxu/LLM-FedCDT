import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from typing import cast

from model_utils import MultiModalNet
from multimodal_dataset import PartitionedMultiModalCDT
from fl_client import FedAvgClient
from fl_server import FedAvgServer

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NUMEXPR_MAX_THREADS'] = '20'


def parse_args():
    p = argparse.ArgumentParser("FedLab MultiModal CDT")
    p.add_argument("--img_dir", type=str,
                   default=r"...",
                   help="图像数据目录")
    p.add_argument("--label_txt", type=str,
                   default=r"...",
                   help="标签文件路径")
    p.add_argument("--desc_xlsx", type=str,
                   default=r"...",
                   help="文本描述文件路径（保持兼容性）")
    p.add_argument("--bert_model", type=str, default=r"...", help="BERT模型名称")

    p.add_argument("--total_clients", type=int, default=10, help="客户端总数")
    p.add_argument("--com_round", type=int, default=50, help="联邦学习的总通信轮次")
    p.add_argument("--sample_ratio", type=float, default=1.0, help="每轮参与训练的客户端比例")
    p.add_argument("--batch_size", type=int, default=32, help="批次大小")
    p.add_argument("--epochs", type=int, default=2, help="单轮通信中对本地数据的训练轮次")
    p.add_argument("--lr", type=float, default=5e-5, help="学习率")
    p.add_argument("--max_len", type=int, default=128, help="BERT文本最大长度")
    p.add_argument("--gpu", action="store_true", help="是否使用 GPU")
    p.add_argument("--alpha", type=float, default=0.7, help="Dirichlet分布参数，控制数据非IID程度")
    p.add_argument("--output_dir", type=str, default="./output", help="结果输出目录")
    return p.parse_args()


def evaluate_metrics(model, dataloader, loss_fn, device, num_classes):
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


class EvalPipeline(StandalonePipeline):
    def __init__(self, server_handler, client_trainer, test_loader, loss_fn, device, num_classes, metrics_csv_path,
                 total_rounds):
        super().__init__(server_handler, client_trainer)
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.metrics_history = []
        self.metrics_csv_path = metrics_csv_path
        self.total_rounds = total_rounds

    def main(self):
        handler: SyncServerHandler = cast(SyncServerHandler, self.handler)
        round_idx = 0
        while not handler.if_stop:
            sampled = handler.sample_clients()
            downlink = handler.downlink_package
            self.trainer.local_process(downlink, sampled)
            uploads = self.trainer.uplink_package
            for pack in uploads:
                handler.load(pack)

            round_idx += 1

            model_to_eval = handler._model.to(self.device)

            metrics = evaluate_metrics(model_to_eval, self.test_loader, self.loss_fn, self.device,
                                       self.num_classes)
            metrics["round"] = round_idx
            self.metrics_history.append(metrics)

            print(f"=== Round {round_idx}/{self.total_rounds} ===")
            print(f"  Test Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision (weighted): {metrics['precision']:.4f}")
            print(f"  Recall (weighted): {metrics['recall']:.4f}")
            print(f"  F1-score (weighted): {metrics['f1_score']:.4f}")
            print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            print("-" * 30)

        df_metrics = pd.DataFrame(self.metrics_history)
        df_metrics.to_csv(self.metrics_csv_path, index=False)
        print(f"\n所有轮次的评估指标已保存到: {self.metrics_csv_path}")

        if not df_metrics.empty:
            accuracies = df_metrics["accuracy"].values
            max_accuracy = accuracies.max()
            convergence_threshold = 0.95 * max_accuracy

            convergence_round = -1
            for i, acc in enumerate(accuracies):
                if acc >= convergence_threshold:
                    convergence_round = i + 1
                    break

            print(
                f"模型收敛轮数 (Accuracy 达到最高值的 95%): {convergence_round if convergence_round != -1 else '未达到'}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv_path = os.path.join(args.output_dir, "federated_metrics1.csv")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}. Did you forget to use --gpu?")

    print("\n请选择训练模式：")
    print("  [1] 仅使用 ResNet-18（图像）")
    print("  [2] 仅使用 BERT（文本）")
    print("  [3] 使用 ResNet-18 + BERT（多模态）")
    choice = input("请输入1、2或3并回车: ").strip()

    use_image = False
    use_text = False

    if choice == "1":
        use_image = True
        print("==> 已选择模式：仅图像")
    elif choice == "2":
        use_text = True
        print("==> 已选择模式：仅文本")
    elif choice == "3":
        use_image = True
        use_text = True
        print("==> 已选择模式：多模态")
    else:
        print("无效的选择，默认为多模态模式。")
        use_image = True
        use_text = True

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    partitioner = PartitionedMultiModalCDT(
        img_dir=args.img_dir, label_txt=args.label_txt, desc_xlsx=args.desc_xlsx,
        num_clients=args.total_clients, batch_size=args.batch_size, alpha=args.alpha,
        tokenizer_name=args.bert_model, max_length=args.max_len, transform=transform
    )

    num_classes = 5
    model = MultiModalNet(
        num_classes,
        bert_name=args.bert_model,
        use_image=use_image,
        use_text=use_text,
        device=device
    ).to(device)

    server = FedAvgServer(
        model=model, global_rounds=args.com_round, sample_ratio=args.sample_ratio,
        device=device
    )

    client = FedAvgClient(
        model=model, num_clients=args.total_clients, device=device
    )
    client.setup_dataset(partitioner)
    client.setup_optim(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    test_loader = partitioner.get_test_dataloader(args.batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()

    pipeline = EvalPipeline(
        server, client, test_loader, loss_fn, device, num_classes, metrics_csv_path, args.com_round
    )
    pipeline.main()


if __name__ == "__main__":
    main()