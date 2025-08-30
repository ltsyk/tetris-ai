from torch.utils.tensorboard import SummaryWriter


class CustomTensorBoard:
    """Simple wrapper around SummaryWriter for unified logging."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log(self, step, **stats):
        for name, value in stats.items():
            self.writer.add_scalar(name, value, step)

    def close(self):
        self.writer.close()

