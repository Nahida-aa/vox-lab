import argparse
import pathlib
import torch

from torch import nn
from utils import load_yaml_config


class FlowWithSpeedControl(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.flow_module = module
        
    def forward(self,
            token,
            prompt_token,
            prompt_feat,
            embedding,
            speed):
        feat = self.flow_module.forward(
            token,
            prompt_token,
            prompt_feat,
            embedding)
        if speed.item() != 1.0:
            feat = nn.functional.interpolate(feat, size=int(feat.shape[2] / speed), mode='linear')
        return feat


def get_dummy_input_estimator(batch_size=2, seq_len=256, out_channels=80, device=None, dtype=torch.float32):
    x = torch.rand([batch_size, out_channels, seq_len], dtype=dtype, device=device)
    mask = torch.ones([batch_size, 1, seq_len], dtype=dtype, device=device)
    mu = torch.rand([batch_size, out_channels, seq_len], dtype=dtype, device=device)
    t = torch.rand([batch_size], dtype=dtype, device=device)
    spks = torch.rand([batch_size, out_channels], dtype=dtype, device=device)
    cond = torch.rand([batch_size, out_channels, seq_len], dtype=dtype, device=device)
    return x, mask, mu, t, spks, cond


def get_dummy_input_flow(batch_size=1, token_len=100, out_channels=80, prompt_token_len=50, prompt_feat_len=192, embedding_length=192, device=None, token_dtype=torch.int64, prompt_token_dtype=torch.int32, dtype=torch.float32, add_speed_control=False):
    token = torch.randint(0, 100, (batch_size, token_len), dtype=token_dtype, device=device)
    prompt_token = torch.randint(0, 100, (batch_size, prompt_token_len), dtype=prompt_token_dtype, device=device)
    prompt_feat = torch.randn(batch_size, prompt_feat_len, out_channels, dtype=dtype, device=device)
    embedding = torch.randn(batch_size, embedding_length, dtype=dtype, device=device)
    if add_speed_control:
        speed = torch.tensor(1.5, dtype=torch.float32, device=device)
        return token, prompt_token, prompt_feat, embedding, speed
    return token, prompt_token, prompt_feat, embedding


@torch.no_grad()
def run_onnx_export(config_path, flow_path, output_path, device, dtype, int32_token=False, add_speed_control=False):
    flow = load_yaml_config(config_path, 'flow')
    flow.load_state_dict(torch.load(flow_path), strict=True)
    flow.to(device=device, dtype=dtype).eval()

    # First, trace the estimator module
    flow.decoder.estimator = torch.jit.trace(flow.decoder.estimator, get_dummy_input_estimator(device=device, dtype=dtype))
    if add_speed_control:
        flow = FlowWithSpeedControl(flow)
    # Then, script the whole flow module
    flow = torch.jit.script(flow)
    # Finally, export to ONNX
    torch.onnx.export(
        flow,
        get_dummy_input_flow(device=device, token_dtype=torch.int32 if int32_token else torch.int64, dtype=dtype, add_speed_control=add_speed_control),
        output_path,
        input_names=['token','prompt_token','prompt_feat','embedding'] + (['speed'] if add_speed_control else []),
        output_names=['tts_mel'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'token': { 1: "token_len"},
            'prompt_token': {1: "prompt_token_len"},
            'prompt_feat': {1: "prompt_feat_len"},
            'embedding': {1: "embedding_len"},
            'tts_mel': {2: "mel_len"},
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Flow module to ONNX format")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--flow_name', type=str, default='flow.pt', help='Name of the Flow model file (.pt)')
    parser.add_argument('--half', action='store_true', help='Use half precision for module parameters')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the exported ONNX module')
    parser.add_argument('--int32_token', action='store_true', help='Use int32 for token input')
    parser.add_argument('--add_speed_control', action='store_true', help='Add speed control input to the flow module')
    parser.add_argument('--device', type=str, default='default', help='Device to use for ONNX export')
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    config_path = model_path / 'cosyvoice2.yaml'
    flow_path = model_path / args.flow_name
    if not config_path.exists():
        config_path = model_path / 'cosyvoice3.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in {model_path}")

    if args.device == 'default':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if args.half else torch.float32

    run_onnx_export(config_path, flow_path, args.output_path, device, dtype, args.int32_token, args.add_speed_control)


if __name__ == "__main__":
    main()
