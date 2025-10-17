import torch
import torch.optim as optim
import torch.nn as nn
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer


class FedAvgClient(SGDSerialClientTrainer):
    def __init__(self, model, num_clients: int, device: torch.device):
        super().__init__(model=model, num_clients=num_clients, cuda=(device.type == 'cuda'), device=device)
        self.device = device

    def setup_optim(self, epochs: int, batch_size: int, lr: float):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        model = self._model

        model.to(self.device)

        self._optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr
        )
        self._criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(self.epochs):
            for batch_idx, batch in enumerate(train_loader):
                self.train_step(self, batch)

        return self._model.cpu().state_dict(), len(train_loader.dataset)

    def train_step(self, engine, batch):
        model = engine.model
        optimizer = engine._optimizer
        loss_fn = engine._criterion

        img, input_ids, attention_mask, label = batch

        if model.use_image:
            img = img.to(self.device)
        if model.use_text:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        label = label.to(self.device)

        logits = model(img, input_ids, attention_mask)

        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()