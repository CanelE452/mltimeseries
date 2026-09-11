"""Reuse verified F0 inference with the explicit study31 panel loader."""
from experiments.peft_contribution_freeze_v1.panel import Panel
from experiments.peft_overlap_transfer_v1 import f0

if __name__ == '__main__':
    f0.shared.Panel = Panel
    f0.main()
