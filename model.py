import torch
import torch.nn as nn
from transformers import ElectraModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from dataset import NUM_LABELS, LABEL_PAD


class MedDecModel(nn.Module):
    """
    ELECTRA encoder with a per-token classification head.
    Architecture:
        - ELECTRA-base encoder (last hidden state [batch, seq_len, 768])
        - Dropout (0.1) (regularisation)
        - Linear (768, 19) (to obtain logits for each of the 19 label classes)
    """

    def __init__(self, model_name: str = "google/electra-base-discriminator", num_labels: int = NUM_LABELS, dropout: float = 0.1):
        
        super().__init__()
        
        self.encoder = ElectraModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size   # 768 for ELECTRA base
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_labels))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (long): [batch, seq_len] 
            attention_mask (long): [batch, seq_len]  (1 = real token, 0 = pad)
        Returns:
            logits (float): [batch, seq_len, num_labels]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state     # [batch, seq_len, 768]
        
        return self.classifier(sequence_output)         # [batch, seq_len, 19]


def build_training_objects(model: MedDecModel, total_steps: int, lr: float = 4e-5, warmup_ratio: float = 0.1):
    """
    Build loss criterion, AdamW optimizer, and linear-warmup scheduler.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL_PAD)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    return criterion, optimizer, scheduler
