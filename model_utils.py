import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForSequenceClassification
from torchvision.models import ResNet18_Weights
from typing import OrderedDict
import os


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
            raise ValueError("At least one of the image or text modalities needs to be enabled.")

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
                print(f"Warning: The pre-trained BERT model was not found at the path '{pre_trained_bert_path}'.")
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
            raise ValueError("No modal encoder is enabled.")

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
            raise ValueError("No valid modal input.")

        fused_features = self.fusion_mlp(x)
        logits = self.classifier(fused_features)
        return logits


def _serialize_model_parameters_custom(parameters_dict: OrderedDict) -> torch.Tensor:
    flattened_params = []
    for param_name, param_tensor in parameters_dict.items():
        if 'txt_encoder' in param_name:
            continue
        flattened_params.append(param_tensor.view(-1))
    return torch.cat(flattened_params)


def _deserialize_model_parameters_custom(
        serialized_tensor: torch.Tensor,
        model_state_dict_ref: OrderedDict
) -> OrderedDict:
    deserialized_params = model_state_dict_ref.copy()
    offset = 0
    for key, value_ref in model_state_dict_ref.items():
        if 'txt_encoder' in key:
            continue

        num_elements = value_ref.numel()
        param_tensor_flat = serialized_tensor[offset: offset + num_elements]
        if param_tensor_flat.numel() != num_elements:
            raise ValueError(
                f"Mismatch in parameter size for {key}: expected {num_elements}, got {param_tensor_flat.numel()}")

        deserialized_params[key] = param_tensor_flat.view(value_ref.shape)
        offset += num_elements
    return deserialized_params
