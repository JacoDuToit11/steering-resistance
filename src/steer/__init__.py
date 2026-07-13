"""steer — SFT training for resistance to adversarial activation steering.

Core loop: build CAA steering vectors (gated on proven attack efficacy),
LoRA-fine-tune with the injection live in the forward pass toward the model's
own clean answers, evaluate base vs trained under attack on held-out questions.
"""

__version__ = "0.1.0"
