
__all__ = ["count_parameters"]


def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def standardize_batch(self, batch):
    return (batch - self.channel_means) / self.channel_std
