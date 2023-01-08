from transformers import AutoModel
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, Config):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.plm_path)
        self.fc = nn.Bilinear(768, 768, 1)

    def forward(self, batch_inputs_A, batch_inputs_B, batch_inputs_C):
        A = self.bert(input_ids=batch_inputs_A).pooler_output
        B = self.bert(input_ids=batch_inputs_B).pooler_output
        C = self.bert(input_ids=batch_inputs_C).pooler_output
        AB = self.fc(A, B)
        AC = self.fc(A, C)
        out = torch.cat([AB, AC], dim=1)
        return out