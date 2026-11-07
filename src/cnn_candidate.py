"""Verified earlier ResNet18 candidate: validation/inference only, no training.

The checkpoint was trained on the original 21,248 train rows. This adapter
checks its exact membership against the current desktop manifest and returns
probabilities in the competition's official class order.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CHECKPOINT_CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
PROBABILITY_COLUMNS = [0, 1, 2, 4, 5, 3]
EXPECTED_CHECKPOINT_SHA256 = 'bc24cf78b30f87b098805b38ff0ecdd14c8797675e55774ddf36bff71294f33f'
EXPECTED_PREPROCESSING = {
    'image_size': 128, 'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225], 'resize': 'pad_bilinear',
    'exif_transpose': True, 'color_mode': 'RGB',
}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def _canonical(path):
    parts = str(path).replace('\\', '/').split('/')
    if len(parts) < 2 or parts[-2] not in CLASSES or not parts[-1]:
        raise ValueError(f'Invalid labeled image identity: {path}')
    return '/'.join(parts[-2:])


def _checked_checkpoint(project_root):
    root = Path(project_root).resolve()
    path = root / 'models' / 'cnn_resnet18_reference.pt'
    receipt = root / 'models' / 'cnn_resnet18_reference.json'
    manifest = root / 'features' / 'manifest.csv'
    provenance = json.loads(receipt.read_text(encoding='utf-8'))
    digest = _sha256(path)
    if digest != EXPECTED_CHECKPOINT_SHA256 or digest != provenance['checkpoint_sha256']:
        raise ValueError('CNN checkpoint hash differs from the audited original')
    manifest_digest = _sha256(manifest)
    if manifest_digest != provenance['desktop_manifest_sha256']:
        raise ValueError('Desktop manifest has changed since CNN compatibility audit')
    rows = _read_rows(manifest)
    keys = [_canonical(row['path']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate canonical labeled image identities')
    if any(key.split('/')[0] != row['label'] for key, row in zip(keys, rows)):
        raise ValueError('Manifest label does not match canonical image class')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1 or state['classes'] != CHECKPOINT_CLASSES:
        raise ValueError('Unexpected CNN checkpoint format/classes')
    if state['config']['model'] != 'resnet18' or state['config'].get('fold') is not None:
        raise ValueError('Expected original fixed-split ResNet18')
    if state['preprocessing'] != EXPECTED_PREPROCESSING:
        raise ValueError('CNN preprocessing differs from the audited original')
    if state['manifest_sha256'] != provenance['source_manifest_sha256']:
        raise ValueError('Original checkpoint manifest digest mismatch')
    indices = {split: [i for i, row in enumerate(rows) if row['split'] == split]
               for split in ('train', 'val', 'test')}
    if sum(map(len, indices.values())) != len(rows):
        raise ValueError('Unexpected manifest split')
    for split in ('train', 'val'):
        member_list = [_canonical(p) for p in state[f'{split}_paths']]
        expected = {keys[i] for i in indices[split]}
        if len(member_list) != len(expected) or set(member_list) != expected:
            raise ValueError(f'CNN {split} members differ from current split')
    if len(indices['train']) != 21248 or len(indices['val']) != 2656:
        raise ValueError('CNN fixed-split counts changed')
    meta = {
        'candidate': 'cnn_resnet18_reference', 'model': 'resnet18',
        'classes': CLASSES, 'checkpoint_classes': CHECKPOINT_CLASSES,
        'probability_column_permutation': PROBABILITY_COLUMNS,
        'checkpoint': str(path), 'checkpoint_sha256': digest,
        'manifest': str(manifest), 'manifest_sha256': manifest_digest,
        'data_sha256': provenance['image_content_sha256'],
        'checkpoint_epoch': state['epoch'],
        'preprocessing': EXPECTED_PREPROCESSING,
        'train_indices': indices['train'], 'val_indices': indices['val'],
        'train_count': len(indices['train']), 'val_count': len(indices['val']),
        'exact_training_members_verified': True,
        'exact_validation_members_verified': True,
        'weights_only_load': True, 'inference_dtype': 'float32',
        'tta': 'none', 'uses_external_answers': False,
        'original_validation_metrics': state['validation'],
        'note': 'Earlier real run stopped after epoch 8; selected checkpoint is epoch 6. '
                'No additional CNN training or validation refit is performed.',
    }
    return state, rows, meta


class CNNTransform:
    """Exact original train.py image_tensor(..., augment=False) transform."""
    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB')
        image = ImageOps.pad(image, (128, 128), method=Image.Resampling.BILINEAR,
                             color=(128, 128, 128))
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(EXPECTED_PREPROCESSING['mean'])[:, None, None]
        std = tensor.new_tensor(EXPECTED_PREPROCESSING['std'])[:, None, None]
        return (tensor - mean) / std


class _Images(Dataset):
    def __init__(self, rows, transform):
        self.paths = [str(row['path']) for row in rows]
        self.transform = transform
        for path in self.paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image)


def load_cnn_candidate(project_root, device='cuda'):
    """Return (eval model, callable PIL transform, checked metadata)."""
    state, _, meta = _checked_checkpoint(project_root)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly choose CPU if needed')
    model = resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 6))
    model.load_state_dict(state['model_state'], strict=True)
    model.requires_grad_(False).to(device).eval()
    meta['strict_state_dict_load'] = True
    return model, CNNTransform(), meta


@torch.inference_mode()
def predict_cnn_candidate(project_root, rows, device='cuda', batch_size=64, workers=2):
    """Return Nx6 FP32 probabilities in CLASSES order, preserving row order."""
    if batch_size <= 0 or workers < 0:
        raise ValueError('Invalid CNN batch_size/workers')
    rows = list(rows)
    if not rows:
        return np.empty((0, 6), dtype=np.float32)
    model, transform, _ = load_cnn_candidate(project_root, device=device)
    device = torch.device(device)
    loader = DataLoader(_Images(rows, transform), batch_size=batch_size,
                        shuffle=False, num_workers=workers,
                        pin_memory=device.type == 'cuda')
    chunks = []
    try:
        for images in loader:
            images = images.to(device, non_blocking=device.type == 'cuda')
            probabilities = model(images).float().softmax(dim=1)
            chunks.append(probabilities[:, PROBABILITY_COLUMNS].cpu().numpy())
        result = np.concatenate(chunks).astype(np.float32, copy=False)
        if result.shape != (len(rows), 6) or not np.isfinite(result).all():
            raise ValueError('Invalid CNN probability result')
        if not np.allclose(result.sum(axis=1), 1.0, atol=2e-6):
            raise ValueError('CNN probabilities do not sum to one')
        return result
    finally:
        del model, loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _metrics(labels, probabilities):
    matrix = np.zeros((6, 6), dtype=np.int64)
    for true, predicted in zip(labels, probabilities.argmax(axis=1)):
        matrix[int(true), int(predicted)] += 1
    diagonal = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(6), where=denominator > 0)
    return {'accuracy': float(diagonal.sum() / len(labels)),
            'macro_f1': float(f1.mean()), 'count': len(labels),
            'confusion_matrix': matrix.tolist()}


def prepare_cnn_candidate(project_root, output):
    """Compute original validation probabilities only; return saved summary dict.

    This function uses CUDA intentionally. The orchestrator must invoke it after
    MLP/LoRA training has released GPU resources. It never evaluates any test set.
    """
    root, output = Path(project_root).resolve(), Path(output).resolve()
    _, rows, meta = _checked_checkpoint(root)
    code_sha = _sha256(Path(__file__))
    summary_path = output / 'summary.json'
    probability_path = output / 'val_probabilities.npy'
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding='utf-8'))
        for field in ('checkpoint_sha256', 'manifest_sha256', 'data_sha256'):
            if saved.get(field) != meta[field]:
                raise ValueError(f'Existing CNN candidate has different {field}')
        if saved.get('adapter_sha256') != code_sha or saved.get('val_indices') != meta['val_indices']:
            raise ValueError('Existing CNN candidate code or validation membership differs')
        if _sha256(probability_path) != saved['val_probabilities_sha256']:
            raise ValueError('Existing CNN validation probabilities changed')
        return saved
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('CNN output must be empty or contain a complete matching cache')
    val_rows = [rows[i] for i in meta['val_indices']]
    probabilities = predict_cnn_candidate(root, val_rows, device='cuda', batch_size=64, workers=2)
    labels = np.array([CLASSES.index(row['label']) for row in val_rows], dtype=np.int64)
    tmp = probability_path.with_suffix('.npy.tmp')
    with tmp.open('wb') as handle:
        np.save(handle, probabilities, allow_pickle=False)
    tmp.replace(probability_path)
    summary = {
        **meta, 'adapter_sha256': code_sha, 'val_probabilities_path': str(probability_path),
        'val_probabilities_sha256': _sha256(probability_path),
        'summary_path': str(summary_path), 'val_metrics': _metrics(labels, probabilities),
        'validation_only': True, 'test_evaluated': False,
        'selection_note': 'Original validation selected the earlier epoch; these are selection metrics.',
    }
    tmp_summary = summary_path.with_suffix('.json.tmp')
    tmp_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_summary.replace(summary_path)
    return summary
