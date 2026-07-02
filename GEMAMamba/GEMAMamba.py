from .lmamba import VSSM
import torch
from torch import nn


class GEMAMamba(nn.Module):
    def __init__(
        self,
        input_channels=3,
        num_classes=1,
        depths=[2, 2, 2, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        self.network = VSSM(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

    def _align_input_channels(self, x):
        if x.size(1) == self.input_channels:
            return x
        if x.size(1) == 1 and self.input_channels == 3:
            return x.repeat(1, 3, 1, 1)
        if x.size(1) == 3 and self.input_channels == 1:
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}.")

    def forward(self, x):
        logits = self.network(self._align_input_channels(x))
        return torch.sigmoid(logits) if self.num_classes == 1 else logits

    def load_from(self):
        if self.load_ckpt_path is None:
            return

        checkpoint_data = torch.load(self.load_ckpt_path)
        pretrained = checkpoint_data["model"]

        model_state = self.network.state_dict()
        encoder_state = {key: value for key, value in pretrained.items() if key in model_state}
        model_state.update(encoder_state)
        print(
            "Total model_dict: {}, Total pretrained_dict: {}, update: {}".format(
                len(model_state), len(pretrained), len(encoder_state)
            )
        )
        self.network.load_state_dict(model_state)
        print("encoder loaded finished!")

        decoder_remap = {}
        for key, value in pretrained.items():
            if "layers.0" in key:
                decoder_remap[key.replace("layers.0", "layers_up.3")] = value
            elif "layers.1" in key:
                decoder_remap[key.replace("layers.1", "layers_up.2")] = value
            elif "layers.2" in key:
                decoder_remap[key.replace("layers.2", "layers_up.1")] = value
            elif "layers.3" in key:
                decoder_remap[key.replace("layers.3", "layers_up.0")] = value

        model_state = self.network.state_dict()
        decoder_state = {key: value for key, value in decoder_remap.items() if key in model_state}
        model_state.update(decoder_state)
        print(
            "Total model_dict: {}, Total pretrained_dict: {}, update: {}".format(
                len(model_state), len(decoder_remap), len(decoder_state)
            )
        )
        self.network.load_state_dict(model_state)


def _forward_check():
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal GEMAMamba forward check.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = GEMAMamba(
        input_channels=args.input_channels,
        num_classes=args.num_classes,
    ).to(device)
    model.eval()

    x = torch.randn(
        args.batch_size,
        args.input_channels,
        args.image_size,
        args.image_size,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    print(f"device: {device}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"output dtype: {y.dtype}")
    print(f"output min/max: {y.min().item():.6f}/{y.max().item():.6f}")


if __name__ == "__main__":
    _forward_check()
