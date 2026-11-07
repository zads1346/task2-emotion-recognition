"""Lock validation selection before this round's reused internal-test comparison."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('XFORMERS_DISABLED', '1')
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from train_classifier import (CLASSES, load_inputs, augmented_scaled, fit_linear, metrics,
    export_report, sha256, write_json, write_csv, predict_chunked, read_csv)

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2.0.0'
FINAL_FILES = ['selected_recipe.json', 'test_report.json', 'test_confusion.csv', 'test_predictions.csv',
               'test_probabilities.npy', 'submission_probabilities.npy', 'submission.csv',
               'prediction_receipt.json', 'evaluation_started.json', 'evaluation_receipt.json']
TEST_NOTE = ('The original internal-test score was already observed before this improvement round. '
             'This is a reused internal-test comparison, not a previously untouched independent estimate. '
             'This round selected models/weights on validation only and locked them before test evaluation.')


def checked_hash(path, digest):
    if not isinstance(digest, str) or len(digest) != 64 or not Path(path).is_file() or sha256(path) != digest:
        raise ValueError(f'Artifact checksum mismatch: {path}')
    return digest


def receipt_files(folder, receipt, key, required):
    mapping = receipt.get(key, {})
    for name in required:
        checked_hash(Path(folder)/name, mapping.get(name))


def source_hashes(root, skip_cnn):
    names = ['finalize_improvements.py', 'train_classifier.py', 'train_mlp.py', 'train_lora.py']
    if not skip_cnn:
        names.append('cnn_candidate.py')
    return {name: sha256(root/'src'/name) for name in names}


def artifact_entry(checkpoint, probability, receipt):
    return {'checkpoint': str(checkpoint), 'sha256': sha256(checkpoint),
            'validation_probability': str(probability), 'validation_probability_sha256': sha256(probability),
            'producer_receipt': str(receipt), 'producer_receipt_sha256': sha256(receipt)}


def verify_artifact_entry(entry):
    for field, digest in [('checkpoint', 'sha256'), ('validation_probability', 'validation_probability_sha256'),
                          ('producer_receipt', 'producer_receipt_sha256')]:
        checked_hash(entry[field], entry[digest])

def rank(m):
    return m['accuracy'], m['macro_f1'], -m['log_loss']

def prepare_baseline(root):
    metadata, hashes, rows, unlabeled, y, split, features = load_inputs(root/'features')
    output = root/'experiments'/'logistic_reference'
    output.mkdir(parents=True, exist_ok=True)
    config = {'classes': CLASSES, 'C': .01, 'class_weight': None, 'max_iter': 300, 'tol': .001,
              'tta_flip': False, 'input_sha256': hashes,
              'train_indices': split['train'].tolist(), 'val_indices': split['val'].tolist()}
    saved = output/'summary.json'
    if saved.exists():
        receipt = json.loads(saved.read_text(encoding='utf-8'))
        if receipt['config'] != config:
            raise RuntimeError('Reference classifier inputs/config changed')
        for name, digest in receipt['file_sha256'].items():
            if sha256(output/name) != digest:
                raise RuntimeError('Reference classifier output checksum mismatch')
        print('Verified existing train-only logistic reference.', flush=True)
        return metadata, hashes, rows, unlabeled, y, split, features
    with threadpool_limits(limits=4):
        scaler, x, labels = augmented_scaled(features, y, split['train'])
        classifier, fit_info = fit_linear(x, labels, .01, None, 300, .001)
        scaler.copy = True
        model = Pipeline([('scaler', scaler), ('classifier', classifier)])
        del x, labels
        probability = predict_chunked(model, features['labeled_original'], features['labeled_flip'],
                                      split['val'], tta=False)
    joblib.dump(model, output/'classifier.joblib', compress=3)
    np.save(output/'val_probabilities.npy', probability.astype(np.float32), allow_pickle=False)
    receipt = {'config': config, 'val_metrics': metrics(y[split['val']], probability), 'fit': fit_info,
               'purpose': 'Validation-safe reference trained on train only; original train+val classifier is not used.',
               'file_sha256': {n: sha256(output/n) for n in ['classifier.joblib', 'val_probabilities.npy']}}
    write_json(saved, receipt)
    print(json.dumps({'reference_val_metrics': receipt['val_metrics']}), flush=True)
    return metadata, hashes, rows, unlabeled, y, split, features

def check_probability(p, n):
    if p.shape != (n, 6) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid candidate probabilities')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('Candidate probabilities do not sum to one')
    # FP32 softmax sums can differ from 1 by ~1e-7. Normalize in FP64 before
    # log_loss or weighted aggregation rather than propagating rounding errors.
    result = np.asarray(p, dtype=np.float64).copy()
    result /= result.sum(axis=1, keepdims=True)
    return result

def metadata_indices(meta, field):
    if field in meta:
        return meta[field]
    if field in meta.get('provenance', {}):
        return meta['provenance'][field]
    raise ValueError(f'Candidate checkpoint lacks {field}')

def check_checkpoint(meta, split, rows, manifest_sha):
    if meta.get('classes') != CLASSES:
        raise ValueError('Candidate class order mismatch')
    for field, name in [('train_indices', 'train'), ('val_indices', 'val')]:
        if list(metadata_indices(meta, field)) != split[name].tolist():
            raise ValueError(f'Candidate {field} does not match the fixed manifest')
    digests = [value for value in [meta.get('manifest_sha256'),
               meta.get('provenance', {}).get('manifest_sha256'),
               meta.get('config', {}).get('input_sha256', {}).get('manifest.csv')] if value is not None]
    if not digests or any(value != manifest_sha for value in digests):
        raise ValueError('Candidate checkpoint manifest hash mismatch')
    canonical = lambda p: os.path.normcase(str(Path(p).resolve()))
    for name in ['train', 'val']:
        expected = [canonical(rows[i]['path']) for i in split[name]]
        actual = [canonical(p) for p in metadata_indices(meta, name+'_paths')]
        if actual != expected or len(actual) != len(set(actual)):
            raise ValueError(f'Candidate exact {name} paths/order mismatch')


def check_submission(output, unlabeled):
    rows = read_csv(output/'submission.csv', ['ID', 'Emotion'])
    if [r['ID'] for r in rows] != [r['ID'] for r in unlabeled]:
        raise ValueError('Submission row count or ID order mismatch')
    if any(r['Emotion'] not in {str(i) for i in range(6)} for r in rows):
        raise ValueError('Submission uses invalid official emotion codes')


def reuse_completed(root, output, skip_cnn):
    """Read and verify receipts only; never recompute metrics on a completed run."""
    saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if saved.get('status') != 'complete' or saved.get('version') != VERSION:
        raise RuntimeError('Existing final result has an unsupported or incomplete receipt')
    receipt_files(output, saved, 'file_sha256', FINAL_FILES)
    recipe = json.loads((output/'selected_recipe.json').read_text(encoding='utf-8'))
    _, hashes, _, unlabeled, _, split, _ = load_inputs(root/'features')
    if recipe['input_sha256'] != hashes or recipe['source_sha256'] != source_hashes(root, skip_cnn):
        raise RuntimeError('Completed final result inputs or source changed')
    if recipe['skip_cnn'] != skip_cnn or saved['selected'] != recipe['selected']:
        raise RuntimeError('Completed final result recipe/config changed')
    if saved['recipe_sha256'] != sha256(output/'selected_recipe.json'):
        raise RuntimeError('Completed final recipe checksum mismatch')
    for entry in recipe['artifacts'].values():
        verify_artifact_entry(entry)
    prediction_receipt = json.loads((output/'prediction_receipt.json').read_text(encoding='utf-8'))
    evaluation_receipt = json.loads((output/'evaluation_receipt.json').read_text(encoding='utf-8'))
    if prediction_receipt['recipe_sha256'] != saved['recipe_sha256'] or evaluation_receipt['recipe_sha256'] != saved['recipe_sha256']:
        raise ValueError('Prediction/evaluation receipts belong to another recipe')
    if saved['test_samples'] != len(split['test']) or saved['submission_rows'] != len(unlabeled):
        raise ValueError('Completed output sample counts changed')
    report = json.loads((output/'test_report.json').read_text(encoding='utf-8'))
    if (saved['test_metrics'] != report['metrics'] or saved['source_sha256'] != recipe['source_sha256'] or
            saved['submission_sha256'] != saved['file_sha256']['submission.csv']):
        raise ValueError('Completed summary differs from the verified reports/recipe')
    check_submission(output, unlabeled)
    print('Verified completed inputs, sources, candidates, recipe and every final artifact; evaluation was not repeated.', flush=True)
    return saved

def predict_lora(root, checkpoint, rows):
    from train_lora import load_lora_checkpoint, make_eval_transform
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    model, meta = load_lora_checkpoint(checkpoint, root, 'cuda')
    model.eval()
    transform = make_eval_transform()
    # workers=0 here avoids local-class Windows spawn issues; training has its own workers.
    class Images(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            with Image.open(rows[i]['path']) as image:
                return transform(image.convert('RGB'))
    output = []
    with torch.inference_mode():
        for batch in DataLoader(Images(), batch_size=16, shuffle=False, num_workers=0, pin_memory=True):
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(batch.cuda(non_blocking=True))
            output.append(logits.float().softmax(dim=1).cpu().numpy())
    del model
    gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--skip-cnn', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    torch.set_num_threads(4)
    output = root/'improved_results'
    if (output/'summary.json').exists():
        reuse_completed(root, output, args.skip_cnn)
        return
    if (output/'evaluation_started.json').exists() and not (output/'evaluation_receipt.json').exists():
        raise RuntimeError('An evaluation was started without a completion receipt. Preserve these files for review; '
                           'automatic rerun would risk repeating the internal-test evaluation.')
    metadata, hashes, rows, unlabeled, y, split, features = prepare_baseline(root)
    if args.baseline_only:
        return
    output.mkdir(parents=True, exist_ok=True)
    candidates = {}
    artifacts = {}
    baseline_dir = root/'experiments'/'logistic_reference'
    baseline_receipt_path = baseline_dir/'summary.json'
    baseline_receipt = json.loads(baseline_receipt_path.read_text(encoding='utf-8'))
    receipt_files(baseline_dir, baseline_receipt, 'file_sha256', ['classifier.joblib', 'val_probabilities.npy'])
    candidates['logistic'] = np.load(baseline_dir/'val_probabilities.npy', allow_pickle=False)
    artifacts['logistic'] = artifact_entry(baseline_dir/'classifier.joblib', baseline_dir/'val_probabilities.npy', baseline_receipt_path)
    from train_mlp import load_mlp_checkpoint
    mlp_path = root/'experiments'/'mlp'/'best.pt'
    mlp_receipt_path = mlp_path.parent/'summary.json'
    mlp_receipt = json.loads(mlp_receipt_path.read_text(encoding='utf-8'))
    if mlp_receipt.get('status') != 'complete' or mlp_receipt.get('tta_flip') is not False:
        raise ValueError('MLP producer must be complete with original-only validation')
    receipt_files(mlp_path.parent, mlp_receipt, 'artifact_sha256',
                  ['config.json', 'best.pt', 'history.csv', 'val_original.npy', 'val_flip.npy',
                   'val_predictions.npy', 'val_predictions.csv', 'val_metadata.json'])
    mlp_model, mlp_meta = load_mlp_checkpoint(mlp_path, 'cpu')
    check_checkpoint(mlp_meta, split, rows, hashes['manifest.csv'])
    if mlp_meta['config']['source_sha256'] != sha256(root/'src'/'train_mlp.py'):
        raise ValueError('MLP checkpoint producer source changed')
    if mlp_meta['config']['input_sha256'] != hashes:
        raise ValueError('MLP checkpoint feature provenance differs')
    mlp_val_meta = json.loads((mlp_path.parent/'val_metadata.json').read_text(encoding='utf-8'))
    if (mlp_val_meta['manifest_sha256'] != hashes['manifest.csv'] or
            mlp_val_meta['manifest_indices'] != split['val'].tolist() or
            mlp_val_meta['best_checkpoint_sha256'] != sha256(mlp_path)):
        raise ValueError('MLP validation probability provenance differs')
    del mlp_model
    candidates['mlp'] = np.load(mlp_path.parent/'val_predictions.npy', allow_pickle=False)
    if not np.array_equal(candidates['mlp'], np.load(mlp_path.parent/'val_original.npy', allow_pickle=False)):
        raise ValueError('MLP default predictions are not the recorded original view')
    artifacts['mlp'] = artifact_entry(mlp_path, mlp_path.parent/'val_predictions.npy', mlp_receipt_path)
    lora_path = root/'experiments'/'lora'/'best.pt'
    lora_receipt_path = lora_path.parent/'summary.json'
    lora_receipt = json.loads(lora_receipt_path.read_text(encoding='utf-8'))
    checked_hash(lora_path, lora_receipt.get('best_checkpoint_sha256'))
    checked_hash(lora_path.parent/'val_probabilities.npy', lora_receipt.get('val_probabilities_sha256'))
    if lora_receipt.get('validation_tta') is not False or lora_receipt.get('test_evaluated') is not False:
        raise ValueError('LoRA producer must use original-only validation without test evaluation')
    lora_meta = torch.load(lora_path, map_location='cpu', weights_only=True)
    check_checkpoint(lora_meta, split, rows, hashes['manifest.csv'])
    if lora_meta['provenance'] != lora_receipt['provenance']:
        raise ValueError('LoRA producer/checkpoint provenance differs')
    if lora_meta['provenance']['source_sha256'] != sha256(root/'src'/'train_lora.py'):
        raise ValueError('LoRA checkpoint producer source changed')
    candidates['lora'] = np.load(lora_path.parent/'val_probabilities.npy', allow_pickle=False)
    artifacts['lora'] = artifact_entry(lora_path, lora_path.parent/'val_probabilities.npy', lora_receipt_path)
    if not args.skip_cnn:
        from cnn_candidate import prepare_cnn_candidate
        cnn_info = prepare_cnn_candidate(root, root/'experiments'/'cnn_reference')
        if list(cnn_info['val_indices']) != split['val'].tolist():
            raise ValueError('CNN validation order mismatch')
        if (cnn_info['classes'] != CLASSES or cnn_info['manifest_sha256'] != hashes['manifest.csv'] or
                list(cnn_info['train_indices']) != split['train'].tolist()):
            raise ValueError('CNN class/manifest/train membership mismatch')
        checked_hash(cnn_info['val_probabilities_path'], cnn_info['val_probabilities_sha256'])
        candidates['cnn'] = np.load(cnn_info['val_probabilities_path'], allow_pickle=False)
        cnn_path = root/'models'/'cnn_resnet18_reference.pt'
        checked_hash(cnn_path, cnn_info['checkpoint_sha256'])
        artifacts['cnn'] = artifact_entry(cnn_path, Path(cnn_info['val_probabilities_path']), root/'experiments'/'cnn_reference'/'summary.json')
    evaluated = []
    for name, probability in candidates.items():
        probability = check_probability(probability, len(split['val']))
        candidates[name] = probability
        evaluated.append({'name': name, 'weights': {name: 1.0}, **metrics(y[split['val']], probability)})
    best_dino = max([r for r in evaluated if r['name'] != 'cnn'], key=rank)
    if 'cnn' in candidates:
        for weight in [.25, .5, .75]:
            weights = {best_dino['name']: weight, 'cnn': 1-weight}
            probability = check_probability(sum(candidates[k]*v for k, v in weights.items()), len(split['val']))
            evaluated.append({'name': f'{best_dino["name"]}+cnn_{weight:.2f}', 'weights': weights,
                              **metrics(y[split['val']], probability)})
    winner = max(evaluated, key=rank)
    recipe = {'version': VERSION, 'classes': CLASSES, 'selection_metric': 'accuracy', 'secondary_metric': 'macro_f1',
              'selected': winner, 'validation_candidates': evaluated, 'artifacts': artifacts,
              'input_sha256': hashes, 'external_answers_used': False,
              'training_splits': ['train'], 'validation_split': 'val', 'heldout_split': 'test',
              'tta_flip': False, 'skip_cnn': args.skip_cnn, 'source_sha256': source_hashes(root, args.skip_cnn),
              'internal_test_previously_observed': True, 'test_interpretation': TEST_NOTE}
    recipe_path = output/'selected_recipe.json'
    if recipe_path.exists():
        if json.loads(recipe_path.read_text(encoding='utf-8')) != recipe:
            raise RuntimeError('A different final recipe was already locked; refusing to change it')
    else:
        write_json(recipe_path, recipe)
    print('Locked validation-only recipe: ' + json.dumps(winner), flush=True)
    # No new internal-test metric has been computed in this improvement round.
    recipe_sha = sha256(recipe_path)
    for entry in artifacts.values():
        verify_artifact_entry(entry)
    test_count = len(split['test'])
    combined = [rows[i] for i in split['test']] + unlabeled
    prediction_receipt_path = output/'prediction_receipt.json'
    if prediction_receipt_path.exists():
        prediction_receipt = json.loads(prediction_receipt_path.read_text(encoding='utf-8'))
        if prediction_receipt['recipe_sha256'] != recipe_sha:
            raise ValueError('Cached final probabilities belong to a different locked recipe')
        receipt_files(output, prediction_receipt, 'file_sha256', ['test_probabilities.npy', 'submission_probabilities.npy'])
        test_probability = check_probability(np.load(output/'test_probabilities.npy', allow_pickle=False), test_count)
        submission_probability = check_probability(np.load(output/'submission_probabilities.npy', allow_pickle=False), len(unlabeled))
        print('Reused checksummed final probability cache.', flush=True)
    else:
        if (output/'evaluation_receipt.json').exists():
            raise RuntimeError('Evaluation receipt exists without its probability receipt')
        probability = np.zeros((len(combined), 6), dtype=np.float64)
        for name, weight in winner['weights'].items():
            checkpoint = Path(artifacts[name]['checkpoint'])
            checked_hash(checkpoint, artifacts[name]['sha256'])
            print(f'Final inference: {name}, weight={weight}', flush=True)
            if name == 'logistic':
                model = joblib.load(checkpoint)
                p_test = predict_chunked(model, features['labeled_original'], features['labeled_flip'], split['test'], False)
                p_submit = predict_chunked(model, features['unlabeled_original'], features['unlabeled_flip'], tta=False)
                p = np.concatenate([p_test, p_submit])
                del model
            elif name == 'mlp':
                from train_mlp import predict_mlp
                model, _ = load_mlp_checkpoint(checkpoint, 'cuda')
                p = np.concatenate([predict_mlp(model, features['labeled_original'], split['test']),
                                    predict_mlp(model, features['unlabeled_original'])])
                del model
                gc.collect(); torch.cuda.empty_cache()
            elif name == 'lora':
                p = predict_lora(root, checkpoint, combined)
            else:
                from cnn_candidate import predict_cnn_candidate
                p = predict_cnn_candidate(root, combined, device='cuda', batch_size=64, workers=2)
            probability += weight * check_probability(p, len(combined))
        probability = check_probability(probability, len(combined))
        test_probability, submission_probability = probability[:test_count], probability[test_count:]
        # Float64 cache reproduces exactly the probabilities used for metrics.
        np.save(output/'test_probabilities.npy', test_probability, allow_pickle=False)
        np.save(output/'submission_probabilities.npy', submission_probability, allow_pickle=False)
        prediction_receipt = {'recipe_sha256': recipe_sha, 'dtype': 'float64', 'classes': CLASSES,
                              'test_rows': test_count, 'submission_rows': len(unlabeled),
                              'file_sha256': {name: sha256(output/name) for name in
                                             ['test_probabilities.npy', 'submission_probabilities.npy']}}
        write_json(prediction_receipt_path, prediction_receipt)
    evaluation_receipt_path = output/'evaluation_receipt.json'
    evaluation_marker_path = output/'evaluation_started.json'
    report_names = ['test_report.json', 'test_confusion.csv', 'test_predictions.csv']
    if evaluation_receipt_path.exists():
        evaluation_receipt = json.loads(evaluation_receipt_path.read_text(encoding='utf-8'))
        if evaluation_receipt['recipe_sha256'] != recipe_sha:
            raise ValueError('Existing evaluation belongs to a different recipe')
        checked_hash(prediction_receipt_path, evaluation_receipt['prediction_receipt_sha256'])
        checked_hash(evaluation_marker_path, evaluation_receipt['evaluation_marker_sha256'])
        receipt_files(output, evaluation_receipt, 'file_sha256', report_names)
        report = json.loads((output/'test_report.json').read_text(encoding='utf-8'))
        print('Reused completed evaluation receipt; no metric recalculation.', flush=True)
    else:
        # Exclusive marker avoids repeating evaluation after interruption or from
        # concurrent processes. A started-but-unreceipted evaluation needs review.
        with evaluation_marker_path.open('x', encoding='utf-8') as handle:
            json.dump({'recipe_sha256': recipe_sha, 'prediction_receipt_sha256': sha256(prediction_receipt_path),
                       'test_interpretation': TEST_NOTE}, handle, indent=2)
        report = export_report(output, 'test', y[split['test']], test_probability, [rows[i] for i in split['test']])
        report['test_interpretation'] = TEST_NOTE
        write_json(output/'test_report.json', report)
        evaluation_receipt = {'recipe_sha256': recipe_sha, 'prediction_receipt_sha256': sha256(prediction_receipt_path),
                              'evaluation_marker_sha256': sha256(evaluation_marker_path),
                              'test_interpretation': TEST_NOTE,
                              'file_sha256': {name: sha256(output/name) for name in report_names}}
        write_json(evaluation_receipt_path, evaluation_receipt)
    submission = [{'ID': row['ID'], 'Emotion': int(label)} for row, label in zip(unlabeled, submission_probability.argmax(1))]
    write_csv(output/'submission.csv', submission, ['ID', 'Emotion'])
    check_submission(output, unlabeled)
    summary = {'version': VERSION, 'status': 'complete', 'selected': winner, 'validation_candidates': evaluated,
               'test_metrics': report['metrics'], 'test_samples': test_count, 'submission_rows': len(submission),
               'submission_path': str(output/'submission.csv'), 'submission_sha256': sha256(output/'submission.csv'),
               'recipe_sha256': recipe_sha, 'external_answers_used': False,
               'source_sha256': recipe['source_sha256'], 'internal_test_previously_observed': True,
               'file_sha256': {name: sha256(output/name) for name in FINAL_FILES},
               'training_samples_per_candidate': len(split['train']),
               'note': TEST_NOTE}
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
