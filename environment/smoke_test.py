import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description='Standalone dependency and CUDA kernel checks; no dataset or model weights required.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'base_prefix': sys.base_prefix, 'platform': platform.platform(), 'checks': [],
              'scope': 'Dependency imports, versions, and synthetic CUDA forward/backward.'}

    def check(name, fn):
        start = time.monotonic()
        try:
            detail = fn()
            record = {'name': name, 'passed': True, 'detail': detail}
        except Exception:
            record = {'name': name, 'passed': False, 'error': traceback.format_exc()}
        record['seconds'] = round(time.monotonic() - start, 3)
        report['checks'].append(record)
        print(json.dumps(record), flush=True)

    def versions():
        assert sys.version_info[:2] == (3, 10), 'Python 3.10 required'
        prefix = Path(sys.prefix).resolve()
        if (prefix / 'conda-meta').is_dir():
            assert Path(os.environ.get('CONDA_PREFIX', '')).resolve() == prefix, 'Activate this Conda environment'
            report['environment_type'] = 'conda'
        else:
            assert sys.prefix != sys.base_prefix, 'Run in a dedicated virtual environment'
            config = (prefix / 'pyvenv.cfg').read_text().lower()
            assert 'include-system-site-packages = false' in config
            report['environment_type'] = 'venv'
        actual = {}
        for entry in json.loads((root / 'DOWNLOAD_MANIFEST.json').read_text())['files']:
            version = importlib.metadata.version(entry['name'])
            assert version == entry['version'], (entry['name'], version, entry['version'])
            dist = importlib.metadata.distribution(entry['name'])
            assert Path(dist.locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()), entry['name']
            actual[entry['name']] = version
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return actual

    check('isolated_pinned_dependencies', versions)
    for name in ['numpy', 'scipy', 'yaml', 'torch', 'torchvision', 'accelerate',
                 'x_transformers', 'gateloop_transformer', 'spconv.pytorch',
                 'torch_scatter', 'pytorch3d', 'flash_attn']:
        check('import:' + name, lambda name=name: str(importlib.import_module(name).__file__))

    if any(not item['passed'] and item['name'] == 'import:torch' for item in report['checks']):
        report['passed'] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        return 1
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261104)

    def device():
        assert torch.cuda.is_available()
        assert torch.cuda.is_bf16_supported()
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cuda_runtime'] = torch.version.cuda
        return {'name': report['gpu'], 'cuda': torch.version.cuda, 'bf16': True}

    check('cuda_device', device)
    if not report['checks'][-1]['passed']:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report['passed'] = False
        args.output.write_text(json.dumps(report, indent=2))
        return 1

    def finite_gradient(value):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0

    def bf16_step():
        model = torch.nn.Linear(64, 32).cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.weight.detach().clone()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(torch.randn(16, 64, device='cuda'))
            assert result.dtype == torch.bfloat16
            loss = result.float().square().mean()
        loss.backward()
        finite_gradient(model.weight)
        optim.step()
        assert not torch.equal(before, model.weight)
        torch.cuda.synchronize()
        return {'loss': loss.item()}

    def flash():
        from flash_attn import flash_attn_qkvpacked_func
        qkv = torch.randn(2, 64, 3, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)
        output.float().square().mean().backward()
        finite_gradient(qkv)
        torch.cuda.synchronize()

    def scatter():
        from torch_scatter import scatter_add
        x = torch.randn(8, 4, device='cuda', requires_grad=True)
        index = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3], device='cuda')
        actual = scatter_add(x, index, dim=0, dim_size=4)
        expected = torch.zeros(4, 4, device='cuda').index_add(0, index, x)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        finite_gradient(x)

    def sparse():
        import spconv.pytorch as spconv
        coords = torch.tensor([[0, x, y, z] for x in range(3) for y in range(3) for z in range(3)], device='cuda', dtype=torch.int32)
        features = torch.randn(len(coords), 8, device='cuda', requires_grad=True)
        layer = spconv.SubMConv3d(8, 8, 3, padding=1, bias=False).cuda()
        result = layer(spconv.SparseConvTensor(features, coords, [4, 4, 4], 1))
        result.features.square().mean().backward()
        finite_gradient(features)
        finite_gradient(layer.weight)
        torch.cuda.synchronize()

    def knn():
        from pytorch3d.ops import knn_points
        points = torch.randn(2, 32, 3, device='cuda', requires_grad=True)
        targets = torch.randn(2, 48, 3, device='cuda')
        distances = knn_points(points, targets, K=1).dists
        expected = torch.cdist(points, targets).square().min(-1).values
        torch.testing.assert_close(distances[..., 0], expected, atol=1e-5, rtol=1e-4)
        distances.mean().backward()
        finite_gradient(points)

    def transformer():
        from x_transformers import PrefixDecoder
        module = importlib.import_module('x_transformers.x_transformers')
        patch = json.loads((root / 'patches/x-transformers-2.3.12.json').read_text())
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == patch['patched_sha256']
        decoder = PrefixDecoder(dim=64, depth=2, heads=2, attn_dim_head=64,
                                attn_num_mem_kv=4, cross_attend=True, cross_attn_num_mem_kv=4,
                                cross_attn_dim_context=64, attn_flash=True,
                                cross_flash_pattern=(True, False)).cuda()
        tokens = torch.randn(2, 8, 64, device='cuda', requires_grad=True)
        context = torch.randn(2, 12, 64, device='cuda', requires_grad=True)
        selector_logits = torch.randn(2, 1, 5, 12, device='cuda', requires_grad=True)
        # Match models/prompt.py: the caller pads prefix rows and memory KV columns.
        bias = torch.nn.functional.pad(2 * selector_logits.sigmoid(), (4, 0, 3, 0))
        kwargs = dict(context=context, prefix_attn_len=3,
                      context_mask=torch.ones(2, 12, device='cuda', dtype=torch.bool))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            guided = decoder(tokens, cross_attn_bias=bias, **kwargs)
            plain = decoder(tokens, **kwargs)
            assert not torch.allclose(guided, plain), 'Bias has no observable effect'
            loss = guided[..., 0].float().square().mean()
        loss.backward()
        finite_gradient(tokens)
        finite_gradient(context)
        finite_gradient(selector_logits)
        decoder.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for length in range(4, 9):
                out = decoder(tokens[:, :length], cross_attn_bias=bias[:, :, :length], **kwargs)
                assert out.shape == (2, length, 64) and torch.isfinite(out).all()
        torch.cuda.synchronize()
        return {'memory_kv': 4, 'selector_logits_gradient_norm': selector_logits.grad.norm().item(),
                'synthetic_prefix_lengths': [4, 5, 6, 7, 8], 'flash_pattern': [True, False]}

    for name, fn in [('bf16_optimizer_step', bf16_step), ('flash_attention_backward', flash),
                     ('scatter_cuda_backward', scatter), ('sparse_convolution_backward', sparse),
                     ('pytorch3d_knn_backward', knn), ('patched_prefix_decoder_bias_backward', transformer)]:
        check(name, fn)
    report['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    report['passed'] = all(c['passed'] for c in report['checks'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
