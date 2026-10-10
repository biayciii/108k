"""Centralized DETR then FedAvg on one manifest and one required pretrained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--init-checkpoint', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('initial.pth', 'centralized', 'fedavg')):
        raise ValueError('Use a new comparison output directory')
    import torch
    from src.common.data.manifest import read_manifest
    from src.common.data.partition import partition_by_gap
    from src.common.utils.config import load_config
    from src.common.utils.seed import set_seed
    from src.engine.trainer import build_trainer

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    doc, info = read_manifest(args.manifest, args.data_root, exploratory=True)
    _, partition = partition_by_gap([r for r in doc['records'] if r['split'] == 'train'], 3)
    config = ROOT / 'configs/detr_comparison.yaml'
    cfg = load_config(config)
    cfg['device'] = 'cpu'
    source = args.init_checkpoint.resolve()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint or 'args' not in checkpoint:
        raise ValueError('Expected full thyroid DETR checkpoint with model and args')
    saved = checkpoint['args']
    saved = saved if isinstance(saved, dict) else vars(saved)
    if saved.get('dataset_file') != 'thyroid' or saved.get('masks', False):
        raise ValueError('Expected thyroid bounding-box checkpoint')
    cfg['data']['brightness_level'] = int(saved['brighness_levels'])
    cfg['architecture']['legacy_args'] = {k: v for k, v in saved.items() if k in {
        'backbone', 'dilation', 'position_embedding', 'enc_layers', 'dec_layers',
        'dim_feedforward', 'hidden_dim', 'dropout', 'nheads', 'num_queries', 'pre_norm',
        'aux_loss', 'set_cost_class', 'set_cost_bbox', 'set_cost_giou',
        'bbox_loss_coef', 'giou_loss_coef', 'eos_coef'}}
    cfg['protocol'] = 'exploratory_detr_shared_pretrained_checkpoint'
    cfg['strict_initial_weights'] = True
    cfg['initial_weights'] = str(source)
    del checkpoint
    import yaml
    config = out / 'comparison_config.yaml'
    config.write_text(yaml.safe_dump(dict(cfg)), encoding='utf-8')
    set_seed(42)
    trainer = build_trainer('detr', cfg)
    frozen = [name for name, value in trainer.model.named_parameters() if not value.requires_grad]
    if frozen:
        raise ValueError('Unexpected frozen model parameters in scratch comparison')
    torch.save({'model': trainer.model.state_dict(), 'config': dict(cfg)}, out / 'initial.pth')
    print('PRETRAINED CHECKPOINT LOADED: 100% model tensors', flush=True)
    del trainer
    protocol = dict(manifest_sha256=info['sha256'], initialization_sha256=digest(out / 'initial.pth'),
                    initialization='shared pretrained detector, strict full weight loading',
                    source_checkpoint=str(source), source_sha256=digest(source),
                    config_sha256=digest(config), patient_identity_verified=doc['patient_identity_verified'],
                    epochs=args.epochs, rounds=args.epochs, local_epochs=1, partition=partition,
                    note='Exploratory fine-tuning; prior checkpoint training overlap unknown; not a clean held-out benchmark')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    common = ['--arch', 'detr', '--data-root', str(args.data_root.resolve()),
              '--config', str(config), '--init-weights', str(out / 'initial.pth'),
              '--device', args.device, '--num-workers', '0', '--seed', '42', '--exploratory']
    commands = [
        ('centralized', 'train.py', ['--split-file', str(args.manifest.resolve()), '--epochs', str(args.epochs)]),
        ('fedavg', 'run_federated.py', ['--manifest', str(args.manifest.resolve()), '--method', 'fedavg',
                                      '--rounds', str(args.epochs), '--local-epochs', '1', '--num-clients', '3',
                                      '--covariate', 'gap_days'])]
    for label, script, extra in commands:
        print(f'START {label}: {args.epochs} epochs/rounds', flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *common,
                        '--output-dir', str(out / label), *extra], check=True)
        print(f'COMPLETED {label}: {out / label}', flush=True)
    print(f'COMPLETED COMPARISON: {out}', flush=True)


if __name__ == '__main__':
    main()
