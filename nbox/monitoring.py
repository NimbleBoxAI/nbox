import torch
import torchvision
import torchdrift


def alarm(p_value):
    assert p_value > 0.01, f"Drift alarm! p-value: {p_value * 100:.03f}%"


class ModelMonitor:
    def __init__(
            self,
            drift_detector,
            feature_layer,
            N=20,
            callback=alarm,
            callback_interval=1
    ):
        self.N = N
        base_outputs = drift_detector.base_outputs
        self.drift_detector = drift_detector
        assert base_outputs is not None, "fit drift detector first"
        feature_dim = base_outputs.size(1)
        self.feature_rb = torch.zeros(
            N,
            feature_dim,
            device=base_outputs.device,
            dtype=base_outputs.dtype
        )
        self.have_full_round = False
        self.next_idx = 0
        self.hook = feature_layer.register_forward_hook(self.collect_hook)
        self.counter = 0
        self.callback = callback
        self.callback_interval = callback_interval

    def unhook(self):
        self.hook.remove()

    def collect_hook(self, module, input, output):
        self.counter += 1
        bs = output.size(0)
        if bs > self.N:
            output = output[-self.N:]
            bs = self.N
        output = output.reshape(bs, -1)
        first_part = min(self.N - self.next_idx, bs)
        self.feature_rb[self.next_idx: self.next_idx + first_part] = output[
                                                                     :first_part]
        if first_part < bs:
            self.feature_rb[: bs - first_part] = self.output[first_part:]
        if not self.have_full_round and self.next_idx + bs >= self.N:
            self.have_full_round = True
        self.next_idx = (self.next_idx + bs) % self.N
        if self.callback and self.have_full_round and self.counter % self.callback_interval == 0:
            p_val = self.drift_detector(self.feature_rb)
            self.callback(p_val)


def fit_detector(dataloader: torch.utils.data.DataLoader, model):
    detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
    feature_extractor = model[:-1]  # without the fc layer
    torchdrift.utils.fit(
        dataloader, feature_extractor, detector, num_batches=1)
    return detector
