"""Four fixed validation candidates; locked inference for the refinement round.

Uses only local audited images/checkpoints and original validation predictions.
Does not read external answers, tune temperatures, or overwrite earlier results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

import finalize_improvements as previous
from train_classifier import CLASSES, load_inputs, sha256, write_json, write_csv, read_csv, metrics, export_report

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
REFINEMENT_FORMAT = 'dinov2_lora_refinement_v1'
TEST_NOTE = ('This internal test was already observed in earlier rounds. The result is a reused internal-test '
             'comparison, not an untouched independent estimate. Four fixed candidates were compared on reused '
             'selection validation, and the winning recipe was locked before this round\'s test inference.')
SOURCES = ['finalize_refinement.py', 'refine_lora.py', 'finalize_improvements.py', 'train_lora.py',
           'train_classifier.py', 'train_mlp.py', 'cnn_candidate.py']
OUTPUT_FILES = ['selected_recipe.json', 'test_report.json', 'test_confusion.csv', 'test_predictions.csv',
                'test_probabilities.npy', 'submission_probabilities.npy', 'submission.csv',
                'prediction_receipt.json', 'evaluation_started.json', 'evaluation_receipt.json']


def code_hashes(root):
    return {name: sha256(root/'src'/name) for name in SOURCES}


def add_file(files, path):
    path = Path(path).resolve()
    files[str(path)] = sha256(path)


def verify_files(files, root=None):
    for name, digest in files.items():
        path = Path(name).resolve()
        if root is not None and not path.is_relative_to(Path(root).resolve()):
            raise ValueError('Receipt references a file outside this project')
        previous.checked_hash(path, digest)


def image_key(path):
    parts = str(path).replace('\\', '/').split('/')
    if 'fer_data' not in parts:
        raise ValueError(f'Image is outside the audited FER archive layout: {path}')
    return '/'.join(parts[parts.index('fer_data'):]).casefold()


def verify_image_rows(root, records):
    """Verify source bytes against the original archive audit, without using labels."""
    inventory_path = root/'data'/'original_splits'/'image_inventory.csv'
    inventory = {image_key(row['path']): row['sha256'] for row in read_csv(inventory_path, ['path', 'sha256'])}
    aggregate = hashlib.sha256()
    for row in records:
        key = image_key(row['path'])
        digest = sha256(row['path'])
        if digest != inventory.get(key):
            raise ValueError(f'Image bytes differ from original archive audit: {row["path"]}')
        aggregate.update(key.encode('utf-8'))
        aggregate.update(bytes.fromhex(digest))
    return {'image_rows_sha256': aggregate.hexdigest(), 'inventory_sha256': sha256(inventory_path)}


def make_views(image, transform, family, include_original=True, include_flip=True):
    """Apply the original eval transform once, then flip its tensor width axis."""
    if family == 'cnn':
        rgb = ImageOps.exif_transpose(image).convert('RGB')
    else:
        # Matches original LoRA training: RGB conversion, then its eval transform.
        rgb = image.convert('RGB')
    tensor = transform(rgb)
    views = []
    if include_original:
        views.append(tensor)
    if include_flip:
        views.append(tensor.flip(-1))
    return torch.stack(views)


class ViewImages(Dataset):
    def __init__(self, records, transform, family, original, flip):
        self.records, self.transform, self.family = records, transform, family
        self.original, self.flip = original, flip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['path']) as image:
            return make_views(image, self.transform, self.family, self.original, self.flip)


@torch.inference_mode()
def predict_image_views(root, kind, checkpoint, records, original=True, flip=True, device='cuda', batch_size=16):
    """Return (original or None, flip or None), Nx6 official-order probabilities."""
    if not original and not flip:
        raise ValueError('At least one inference view is required')
    if kind == 'cnn':
        from cnn_candidate import load_cnn_candidate, PROBABILITY_COLUMNS
        model, transform, meta = load_cnn_candidate(root, device)
        if sha256(checkpoint) != meta['checkpoint_sha256'] or meta['classes'] != CLASSES:
            raise ValueError('CNN loader checkpoint/classes differ from locked candidate')
        columns = PROBABILITY_COLUMNS
    else:
        from train_lora import load_lora_checkpoint, make_eval_transform
        if kind == 'refined_lora':
            from refine_lora import load_refinement_checkpoint
            model, meta = load_refinement_checkpoint(checkpoint, root, device)
        elif kind == 'original_lora':
            model, meta = load_lora_checkpoint(checkpoint, root, device)
        else:
            raise ValueError(f'Unsupported image model: {kind}')
        if meta['classes'] != CLASSES:
            raise ValueError('LoRA loader returned a different class order')
        transform, columns = make_eval_transform(), list(range(6))
    model.eval()
    device = torch.device(device)
    loader = DataLoader(ViewImages(records, transform, 'cnn' if kind == 'cnn' else 'lora', original, flip),
                        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    outputs = [[] for _ in range(int(original)+int(flip))]
    try:
        for views in loader:
            # Separate view forwards keep the effective image batch within budget.
            for view_index in range(views.shape[1]):
                x = views[:, view_index].to(device, non_blocking=device.type == 'cuda')
                if kind != 'cnn' and device.type == 'cuda':
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(x)
                else:
                    logits = model(x)
                p = logits.float().softmax(dim=1)[:, columns].cpu().numpy()
                outputs[view_index].append(p)
        arrays = [previous.check_probability(np.concatenate(parts), len(records)) for parts in outputs]
        return (arrays[0] if original else None), (arrays[-1] if flip else None)
    finally:
        del model, loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def prepare_flip_cache(root, rows, indices, artifacts, input_hashes, sources):
    folder = root/'experiments'/'refinement_tta_cache'
    folder.mkdir(parents=True, exist_ok=True)
    val_rows = [rows[i] for i in indices]
    image_receipt = verify_image_rows(root, val_rows)
    signature = {'version': VERSION, 'classes': CLASSES, 'manifest_indices': indices.tolist(),
                 'paths': [row['path'] for row in val_rows], 'input_sha256': input_hashes,
                 'source_sha256': sources, 'image_receipt': image_receipt,
                 'checkpoints': {name: artifacts[name] for name in ['original_lora', 'cnn']},
                 'flip_rule': 'original eval EXIF/RGB/resize-or-pad/normalize, then tensor width.flip(-1); matches refinement validation'}
    signature_sha = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    result, recorded_files = {}, {}
    for kind in ['original_lora', 'cnn']:
        path, receipt_path = folder/f'{kind}_flip.npy', folder/f'{kind}_flip.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['signature_sha256'] != signature_sha or receipt['signature'] != signature:
                raise ValueError('TTA cache source/data/model membership changed')
            previous.checked_hash(path, receipt['probability_sha256'])
            probabilities = np.load(path, allow_pickle=False)
        else:
            print(f'Validation flip inference: {kind}', flush=True)
            _, probabilities = predict_image_views(root, kind, Path(artifacts[kind]['checkpoint']), val_rows,
                                                    original=False, flip=True)
            temporary = path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, probabilities, allow_pickle=False)
            temporary.replace(path)
            write_json(receipt_path, {'signature_sha256': signature_sha, 'signature': signature,
                                     'probability_sha256': sha256(path)})
        result[kind] = previous.check_probability(probabilities, len(indices))
        add_file(recorded_files, path)
        add_file(recorded_files, receipt_path)
    return result, recorded_files, image_receipt


def fixed_candidates(probabilities):
    """Exactly four prespecified alternatives; no fitted weights or temperature."""
    specifications = [
        ('original_baseline', {'original_lora': .5, 'cnn': .5}, False),
        ('original_lora_cnn_tta', {'original_lora': .5, 'cnn': .5}, True),
        ('refined_lora_cnn_tta', {'refined_lora': .5, 'cnn': .5}, True),
        ('refined_lora_cnn_mlp_tta', {'refined_lora': .45, 'cnn': .45, 'mlp': .10}, True),
    ]
    candidates = []
    for name, weights, tta in specifications:
        combined = None
        for model, weight in weights.items():
            original, flip = probabilities[model]
            value = (original+flip)/2 if tta else original
            combined = weight*value if combined is None else combined+weight*value
        candidates.append(({'name': name, 'weights': weights, 'tta_flip': tta},
                           previous.check_probability(combined, len(combined))))
    return candidates


def validate_refinement_receipt(receipt, checkpoint=None):
    if (receipt.get('format') != REFINEMENT_FORMAT or receipt.get('state') != 'complete' or
            receipt.get('epochs_completed') != 4 or receipt.get('classes') != CLASSES or
            receipt.get('validation_tta') is not True or receipt.get('test_labels_used') is not False or
            receipt.get('test_images_read') != 0 or receipt.get('external_answers_used') is not False):
        raise ValueError('Refinement producer must finish exactly four epochs with TTA selection and no test/external answers')
    if checkpoint is not None and (checkpoint.get('format') != REFINEMENT_FORMAT or
                                   checkpoint.get('validation_tta') is not True):
        raise ValueError('Refinement checkpoint format/TTA mode differs from the completed producer')


def load_validation(root, sources):
    metadata, hashes, rows, unlabeled, y, split, features = load_inputs(root/'features')
    template_path = root/'data'/'raw'/'submission.csv'
    template = read_csv(template_path, ['ID', 'Emotion'])
    if [row['ID'] for row in template] != [row['ID'] for row in unlabeled]:
        raise ValueError('Original submission template ID count/order differs from inference manifest')
    old_recipe_path = root/'improved_results'/'selected_recipe.json'
    old_recipe = json.loads(old_recipe_path.read_text(encoding='utf-8'))
    if old_recipe['classes'] != CLASSES or old_recipe['input_sha256'] != hashes:
        raise ValueError('Original baseline recipe inputs/classes changed')
    if old_recipe['selected']['weights'] != {'lora': .5, 'cnn': .5} or old_recipe['tta_flip']:
        raise ValueError('Expected the previously selected original-view 50/50 LoRA/CNN baseline')
    for name, digest in old_recipe['source_sha256'].items():
        previous.checked_hash(root/'src'/name, digest)
    files, artifacts, probabilities = {}, {}, {}
    add_file(files, old_recipe_path)
    add_file(files, template_path)
    for original_name, name in [('lora', 'original_lora'), ('cnn', 'cnn'), ('mlp', 'mlp')]:
        entry = old_recipe['artifacts'][original_name]
        previous.verify_artifact_entry(entry)
        artifacts[name] = {'checkpoint': entry['checkpoint'], 'sha256': entry['sha256']}
        for field in ['checkpoint', 'validation_probability', 'producer_receipt']:
            add_file(files, entry[field])
        original = previous.check_probability(np.load(entry['validation_probability'], allow_pickle=False), len(split['val']))
        if name in ['original_lora', 'mlp']:
            meta = torch.load(entry['checkpoint'], map_location='cpu', weights_only=True)
            previous.check_checkpoint(meta, split, rows, hashes['manifest.csv'])
        if name == 'mlp':
            receipt = json.loads(Path(entry['producer_receipt']).read_text(encoding='utf-8'))
            parent = Path(entry['checkpoint']).parent
            previous.receipt_files(parent, receipt, 'artifact_sha256', ['best.pt', 'val_original.npy', 'val_flip.npy', 'val_predictions.npy'])
            if not np.array_equal(np.load(parent/'val_original.npy'), np.load(parent/'val_predictions.npy')):
                raise ValueError('MLP original-view cache mismatch')
            flip = previous.check_probability(np.load(parent/'val_flip.npy', allow_pickle=False), len(split['val']))
            add_file(files, parent/'val_flip.npy')
        else:
            flip = None
        probabilities[name] = (original, flip)
    from cnn_candidate import _checked_checkpoint
    _, _, cnn_meta = _checked_checkpoint(root)
    if cnn_meta['manifest_sha256'] != hashes['manifest.csv'] or cnn_meta['val_indices'] != split['val'].tolist():
        raise ValueError('CNN exact audited manifest membership mismatch')
    refined = root/'experiments'/'lora_refinement'
    receipt_path = refined/'summary.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    validate_refinement_receipt(receipt)
    for name, field in [('best.pt', 'best_checkpoint_sha256'), ('val_original.npy', 'val_original_sha256'),
                        ('val_flip.npy', 'val_flip_sha256')]:
        previous.checked_hash(refined/name, receipt[field])
        add_file(files, refined/name)
    add_file(files, receipt_path)
    checkpoint = torch.load(refined/'best.pt', map_location='cpu', weights_only=True)
    validate_refinement_receipt(receipt, checkpoint)
    previous.check_checkpoint(checkpoint, split, rows, hashes['manifest.csv'])
    if receipt.get('train_indices') != split['train'].tolist() or receipt.get('val_indices') != split['val'].tolist():
        raise ValueError('Refinement summary train/validation indices differ from manifest')
    provenance = checkpoint['provenance']
    if provenance != receipt['provenance']:
        raise ValueError('Refinement producer/checkpoint provenance mismatch')
    expected_provenance = {'original_model_source_sha256': sources['train_lora.py'],
                           'refinement_source_sha256': sources['refine_lora.py'],
                           'origin_checkpoint_sha256': artifacts['original_lora']['sha256']}
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError('Refinement origin/model source provenance mismatch')
    artifacts['refined_lora'] = {'checkpoint': str(refined/'best.pt'), 'sha256': receipt['best_checkpoint_sha256']}
    probabilities['refined_lora'] = tuple(previous.check_probability(np.load(refined/name, allow_pickle=False), len(split['val']))
                                         for name in ['val_original.npy', 'val_flip.npy'])
    flips, cached_files, image_receipt = prepare_flip_cache(root, rows, split['val'], artifacts, hashes, sources)
    files.update(cached_files)
    for name in ['original_lora', 'cnn']:
        probabilities[name] = (probabilities[name][0], flips[name])
    baseline_p = .5*probabilities['original_lora'][0]+.5*probabilities['cnn'][0]
    baseline_metrics = metrics(y[split['val']], previous.check_probability(baseline_p, len(split['val'])))
    if any(not np.isclose(baseline_metrics[k], old_recipe['selected'][k], atol=1e-9, rtol=1e-9)
           for k in ['accuracy', 'macro_f1', 'log_loss']):
        raise ValueError('Reconstructed original baseline differs from its locked validation metrics')
    return metadata, hashes, rows, unlabeled, y, split, features, probabilities, artifacts, files, image_receipt


def verify_completed(root, output, sources):
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if summary.get('version') != VERSION or summary.get('status') != 'complete':
        raise ValueError('Unsupported/incomplete refinement completion receipt')
    previous.receipt_files(output, summary, 'file_sha256', OUTPUT_FILES)
    recipe = json.loads((output/'selected_recipe.json').read_text(encoding='utf-8'))
    _, hashes, rows, unlabeled, _, split, _ = load_inputs(root/'features')
    if recipe['input_sha256'] != hashes or recipe['source_sha256'] != sources:
        raise ValueError('Completed refinement inputs/source changed')
    verify_files(recipe['producer_files'], root)
    if (summary['selected'] != recipe['selected'] or summary['recipe_sha256'] != sha256(output/'selected_recipe.json') or
            summary['test_samples'] != len(split['test']) or summary['submission_rows'] != len(unlabeled)):
        raise ValueError('Completed refinement recipe/sample count mismatch')
    report = json.loads((output/'test_report.json').read_text(encoding='utf-8'))
    if summary['test_metrics'] != report['metrics']:
        raise ValueError('Refinement summary differs from recorded report')
    previous.check_submission(output, unlabeled)
    print('Completed refinement verified; no inference or evaluation repeated.', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--prepare-only', action='store_true', help='Verify producers and cache validation flip predictions only')
    args = parser.parse_args()
    root = args.project_root.resolve()
    torch.set_num_threads(4)
    sources = code_hashes(root)
    output = root/'refinement_results'
    if (output/'summary.json').exists():
        return verify_completed(root, output, sources)
    if (output/'evaluation_started.json').exists() and not (output/'evaluation_receipt.json').exists():
        raise RuntimeError('Evaluation started without completion receipt; review the preserved run before any repeated evaluation')
    (metadata, hashes, rows, unlabeled, y, split, features, probabilities, artifacts,
     producer_files, val_image_receipt) = load_validation(root, sources)
    if args.prepare_only:
        print('Validation TTA cache prepared; no test or competition inference performed.', flush=True)
        return
    candidates = [{**spec, **metrics(y[split['val']], probability)} for spec, probability in fixed_candidates(probabilities)]
    selected = max(candidates, key=previous.rank)
    recipe = {'version': VERSION, 'classes': CLASSES, 'selection_metric': 'accuracy', 'secondary_metric': 'macro_f1',
              'selected': selected, 'validation_candidates': candidates, 'artifacts': artifacts,
              'producer_files': producer_files, 'input_sha256': hashes, 'source_sha256': sources,
              'val_indices': split['val'].tolist(), 'val_paths': [rows[i]['path'] for i in split['val']],
              'val_image_receipt': val_image_receipt, 'external_answers_used': False,
              'training_splits': ['train'], 'validation_split': 'val', 'internal_test_previously_observed': True,
              'test_interpretation': TEST_NOTE, 'temperatures_fitted': False, 'candidate_count': 4}
    output.mkdir(parents=True, exist_ok=True)
    recipe_path = output/'selected_recipe.json'
    if recipe_path.exists():
        if json.loads(recipe_path.read_text(encoding='utf-8')) != recipe:
            raise ValueError('A different refinement recipe is already locked')
    else:
        write_json(recipe_path, recipe)
    print('Locked refinement recipe: '+json.dumps(selected), flush=True)
    recipe_sha = sha256(recipe_path)
    verify_files(producer_files, root)
    prediction_receipt_path = output/'prediction_receipt.json'
    test_count = len(split['test'])
    combined_rows = [rows[i] for i in split['test']]+unlabeled
    if prediction_receipt_path.exists():
        prediction_receipt = json.loads(prediction_receipt_path.read_text(encoding='utf-8'))
        if prediction_receipt['recipe_sha256'] != recipe_sha:
            raise ValueError('Final prediction cache belongs to a different refinement recipe')
        previous.receipt_files(output, prediction_receipt, 'file_sha256', ['test_probabilities.npy', 'submission_probabilities.npy'])
        test_p = previous.check_probability(np.load(output/'test_probabilities.npy'), test_count)
        submit_p = previous.check_probability(np.load(output/'submission_probabilities.npy'), len(unlabeled))
    else:
        if (output/'evaluation_receipt.json').exists():
            raise RuntimeError('Evaluation receipt exists without prediction receipt')
        inference_images = verify_image_rows(root, combined_rows)
        probability = np.zeros((len(combined_rows), 6), dtype=np.float64)
        for name, weight in selected['weights'].items():
            checkpoint = Path(artifacts[name]['checkpoint'])
            previous.checked_hash(checkpoint, artifacts[name]['sha256'])
            print(f'Final selected inference: {name}, weight={weight}, flip={selected["tta_flip"]}', flush=True)
            if name == 'mlp':
                from train_mlp import load_mlp_checkpoint, predict_mlp
                model, _ = load_mlp_checkpoint(checkpoint, 'cuda')
                original = np.concatenate([predict_mlp(model, features['labeled_original'], split['test']),
                                           predict_mlp(model, features['unlabeled_original'])])
                flipped = np.concatenate([predict_mlp(model, features['labeled_flip'], split['test']),
                                          predict_mlp(model, features['unlabeled_flip'])]) if selected['tta_flip'] else None
                del model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                original, flipped = predict_image_views(root, name, checkpoint, combined_rows,
                                                         original=True, flip=selected['tta_flip'])
            p = (original+flipped)/2 if selected['tta_flip'] else original
            probability += weight*previous.check_probability(p, len(combined_rows))
        probability = previous.check_probability(probability, len(combined_rows))
        test_p, submit_p = probability[:test_count], probability[test_count:]
        np.save(output/'test_probabilities.npy', test_p, allow_pickle=False)
        np.save(output/'submission_probabilities.npy', submit_p, allow_pickle=False)
        write_json(prediction_receipt_path, {'recipe_sha256': recipe_sha, 'inference_image_receipt': inference_images,
                   'file_sha256': {name: sha256(output/name) for name in ['test_probabilities.npy', 'submission_probabilities.npy']}})
    evaluation_path, marker_path = output/'evaluation_receipt.json', output/'evaluation_started.json'
    report_files = ['test_report.json', 'test_confusion.csv', 'test_predictions.csv']
    if evaluation_path.exists():
        receipt = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if receipt['recipe_sha256'] != recipe_sha:
            raise ValueError('Existing refinement evaluation belongs to a different recipe')
        previous.checked_hash(prediction_receipt_path, receipt['prediction_receipt_sha256'])
        previous.checked_hash(marker_path, receipt['evaluation_marker_sha256'])
        previous.receipt_files(output, receipt, 'file_sha256', report_files)
        report = json.loads((output/'test_report.json').read_text(encoding='utf-8'))
    else:
        with marker_path.open('x', encoding='utf-8') as stream:
            json.dump({'recipe_sha256': recipe_sha, 'prediction_receipt_sha256': sha256(prediction_receipt_path)}, stream)
        report = export_report(output, 'test', y[split['test']], test_p, [rows[i] for i in split['test']])
        report['test_interpretation'] = TEST_NOTE
        write_json(output/'test_report.json', report)
        write_json(evaluation_path, {'recipe_sha256': recipe_sha, 'prediction_receipt_sha256': sha256(prediction_receipt_path),
                   'evaluation_marker_sha256': sha256(marker_path),
                   'file_sha256': {name: sha256(output/name) for name in report_files}})
    submission = [{'ID': row['ID'], 'Emotion': int(label)} for row, label in zip(unlabeled, submit_p.argmax(1))]
    write_csv(output/'submission.csv', submission, ['ID', 'Emotion'])
    previous.check_submission(output, unlabeled)
    template = read_csv(root/'data'/'raw'/'submission.csv', ['ID', 'Emotion'])
    if [row['ID'] for row in submission] != [row['ID'] for row in template]:
        raise ValueError('Refinement submission IDs/order do not match the original submission template')
    summary = {'version': VERSION, 'status': 'complete', 'classes': CLASSES, 'selected': selected,
               'validation_candidates': candidates, 'test_metrics': report['metrics'], 'test_samples': test_count,
               'submission_rows': len(submission), 'submission_path': str(output/'submission.csv'),
               'submission_sha256': sha256(output/'submission.csv'), 'recipe_sha256': recipe_sha,
               'source_sha256': sources, 'external_answers_used': False, 'internal_test_previously_observed': True,
               'note': TEST_NOTE, 'file_sha256': {name: sha256(output/name) for name in OUTPUT_FILES}}
    write_json(output/'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'file_sha256'}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    main()
