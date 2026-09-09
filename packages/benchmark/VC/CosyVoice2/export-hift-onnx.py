"""Export CosyVoice2 HiFT (HiFTGenerator) to ONNX.

Usage:
  python export-hift-onnx.py [--output OUTPUT_DIR]

The exported model takes mel spectrogram [1, 80, T] + optional cache_source [1, 1, C]
and outputs waveform [1, 1, T'].
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'CosyVoice'))
sys.path.insert(0, str(REPO_ROOT / 'third_party' / 'CosyVoice' / 'third_party' / 'Matcha-TTS'))

import torch
torch.set_default_dtype(torch.float32)


class HiFTWrapper(torch.nn.Module):
    """Wraps HiFT inference into a single forward pass for ONNX export.

    The cache_source branch is handled by always passing a zero-length cache
    (non-streaming mode), which is the benchmark scenario.
    """

    def __init__(self, hift):
        super().__init__()
        self.f0_predictor = hift.f0_predictor
        self.f0_upsamp = hift.f0_upsamp
        self.m_source = hift.m_source
        self.conv_pre = hift.conv_pre
        self.ups = hift.ups
        self.source_downs = hift.source_downs
        self.source_resblocks = hift.source_resblocks
        self.resblocks = hift.resblocks
        self.conv_post = hift.conv_post
        self.reflection_pad = hift.reflection_pad
        self.num_upsamples = hift.num_upsamples
        self.num_kernels = hift.num_kernels
        self.lrelu_slope = hift.lrelu_slope
        self.audio_limit = hift.audio_limit
        self.istft_params = hift.istft_params
        self.register_buffer('stft_window', hift.stft_window)

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)

        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(speech_feat)
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params['n_fft'] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params['n_fft'] // 2 + 1:, :])
        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params['n_fft'], self.istft_params['hop_len'],
            self.istft_params['n_fft'], window=self.stft_window,
            return_complex=True)
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        out = torch.istft(
            torch.complex(real, img),
            self.istft_params['n_fft'], self.istft_params['hop_len'],
            self.istft_params['n_fft'], window=self.stft_window)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(REPO_ROOT / 'pretrained_models' / 'CosyVoice2-0.5B'))
    args = parser.parse_args()

    from cosyvoice.cli.cosyvoice import CosyVoice2
    model_dir = str(REPO_ROOT / 'pretrained_models' / 'CosyVoice2-0.5B')

    print('Loading CosyVoice2...')
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

    hift_wrapper = HiFTWrapper(cosyvoice.model.hift)
    hift_wrapper.eval()

    dummy_mel = torch.randn(1, 80, 100)

    output_path = os.path.join(args.output, 'hift.onnx')

    print(f'Exporting HiFT to {output_path}...')
    torch.onnx.export(
        hift_wrapper,
        (dummy_mel,),
        output_path,
        input_names=['speech_feat'],
        output_names=['waveform'],
        dynamic_axes={
            'speech_feat': {2: 'T'},
            'waveform': {2: 'T_out'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'Done! Saved to {output_path}')

    # Patch ScatterND int32 indices → int64 (required by ONNX Runtime)
    print('Patching ScatterND int32→int64...')
    import onnx
    from onnx import helper, TensorProto

    m = onnx.load(output_path)
    indices_to_cast = {}
    for node in m.graph.node:
        if node.op_type != 'ScatterND':
            continue
        indices_name = node.input[1]
        is_int32 = False
        for vi in m.graph.value_info:
            if vi.name == indices_name and vi.type.tensor_type.elem_type == TensorProto.INT32:
                is_int32 = True
                break
        if not is_int32:
            for vi in m.graph.input:
                if vi.name == indices_name and vi.type.tensor_type.elem_type == TensorProto.INT32:
                    is_int32 = True
                    break
        if is_int32 and indices_name not in indices_to_cast:
            indices_to_cast[indices_name] = f'{indices_name}_int64'

    if indices_to_cast:
        existing_names = {n.name for n in m.graph.node}
        cast_nodes = []
        for old_name, new_name in indices_to_cast.items():
            first_use_idx = None
            for i, node in enumerate(m.graph.node):
                if old_name in node.input:
                    first_use_idx = i
                    break
            cast_name = 'cast_' + old_name + '_to_int64'
            # ensure unique
            if cast_name in existing_names:
                j = 1
                while f'{cast_name}_{j}' in existing_names:
                    j += 1
                cast_name = f'{cast_name}_{j}'
            cast_node = helper.make_node(
                'Cast', inputs=[old_name], outputs=[new_name],
                name=cast_name, to=TensorProto.INT64)
            cast_nodes.append((first_use_idx, cast_node, old_name, new_name))
            existing_names.add(cast_name)
        for idx, cast_node, old_name, new_name in sorted(cast_nodes, key=lambda x: -x[0]):
            m.graph.node.insert(idx, cast_node)
            for node in m.graph.node:
                if node.op_type == 'ScatterND' and node.input[1] == old_name:
                    node.input[1] = new_name
        for _, _, _, new_name in cast_nodes:
            vi = helper.make_tensor_value_info(new_name, TensorProto.INT64, None)
            m.graph.value_info.append(vi)
        onnx.save(m, output_path)
        print(f'  Patched {len(indices_to_cast)} ScatterND int32→int64')
    else:
        print('  No ScatterND int32 indices found')
    print(f'Done! Model saved to {output_path}')


if __name__ == '__main__':
    main()
