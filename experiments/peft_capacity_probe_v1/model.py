"""A wide ReLU head with exactly the LoRA+small-head parameter count."""
import math
import torch
from torch import nn
from experiments.peft_mechanism_diagnostics_v1.controlled_fit import Controlled as OriginalControlled, parameter_digest


class WideHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(768, 1601, bias=False)
        self.hidden_bias = nn.Parameter(torch.empty(1109))
        nn.init.uniform_(self.hidden_bias, -1/math.sqrt(768), 1/math.sqrt(768))
        self.output = nn.Linear(1601, 336)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden):
        bias = torch.nn.functional.pad(self.hidden_bias, (0, 492))
        return self.output(torch.relu(self.input(hidden) + bias))


class Controlled(OriginalControlled):
    def __init__(self, base, job, channels):
        super().__init__(base, job, channels)
        if job['arm'] == 'WIDE':
            assert not job['blocks'] and base.model_dim == 768
            torch.manual_seed(job['seed'])
            self.probe = WideHead().to(base.device)
            self.initial_head_hash = parameter_digest(self, prefix='probe.')
            self.trainable_names = [n for n,p in self.named_parameters() if p.requires_grad]
            self.trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            assert self.trainable_count == 1768949
