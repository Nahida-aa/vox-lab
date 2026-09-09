import pathlib
import argparse
import torch

from utils import load_yaml_config


def load_hift_state_dict(path, hift, strict=True):
    state_dict = torch.load(path)
    hift_state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items()}
    hift.load_state_dict(hift_state_dict, strict=strict)
    return hift


@torch.no_grad()
def run_onnx_export(config_path, hift_path, output_path, device):
    hift = load_yaml_config(config_path, 'hift')
    hift = load_hift_state_dict(hift_path, hift)
    hift.build_istft()
    hift.to(device).eval()

    torch.onnx.export(
        hift,
        (torch.randn([1, 80, 2000], dtype=torch.float32, device=device),),
        output_path,
        input_names=['speech_feat'],
        output_names=['generated_speech'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'speech_feat': {0: "batch_size", 2: "seq_len"},
            'generated_speech': {0: "batch_size", 1: "seq_len"}
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Convert HiFT module to ONNX format")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--hift_name', type=str, default='hift.pt', help='Name of the HiFT model file (.pt)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the exported ONNX module')
    parser.add_argument('--device', type=str, default='default', help='Device to use for ONNX export')
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    config_path = model_path / 'cosyvoice2.yaml'
    hift_path = model_path / args.hift_name
    if not config_path.exists():
        config_path = model_path / 'cosyvoice3.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in {model_path}")

    if args.device == 'default':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    run_onnx_export(config_path, hift_path, args.output_path, device)


if __name__ == '__main__':
    main()
