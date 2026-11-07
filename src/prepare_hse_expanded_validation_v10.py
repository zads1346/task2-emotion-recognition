"""Lock a new official-initialized HSE baseline against the larger grouped split."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import hse_model as model_api
import train_hse_specialist as h
import prepare_expanded_validation_v10 as data_api
import train_expanded_validation_v10 as trainer

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'hse_expanded_validation_v10'
REQUIRED_SOURCES = (
    'src/prepare_expanded_validation_v10.py', 'src/prepare_hse_expanded_validation_v10.py',
    'src/train_expanded_validation_v10.py', 'src/finalize_expanded_validation_v10.py',
    'src/run_hse_expanded_validation_v10.py', 'src/hse_expanded_validation_v10_worker.py',
    'start_hse_expanded_validation_v10.ps1', 'run_hse_expanded_validation_v10_worker.ps1',
    'src/group_training_images.py', 'src/hse_model.py', 'src/train_hse_specialist.py',
    'src/train_hse_sam_v7.py', 'src/validate_cosine_fusion.py', 'src/run_hse_sam_v7.py',
)
PRESERVED_SUBMISSIONS = {
    'ddamfn_v8_results/submission.csv': '55a037351e0677eadc82ebe81ce6a9c208da402f304cd9eeccba70e2121ce4b6',
    'lora_task_v4_results/submission.csv': 'cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd',
    'lora_zero_weight_v5_results/submission.csv': 'cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd',
    'lora_flip_consistency_v6_results/submission.csv': 'cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd',
    'hse_sam_v7_results/submission.csv': 'cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd',
    'hse_results/submission.csv': 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7',
    'fusion_results/submission.csv': '2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Pinned artifact outside this project')
    digest = h.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Unexpected checksum: ' + str(path))
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError('Conflicting source pin')
    pins[str(path)] = digest


def build_plan(root=ROOT):
    root = Path(root).resolve()
    if root != ROOT:
        raise ValueError('Fixed project root required')
    rows, split = data_api.load_records(root, verify_images=True)
    pins = {}
    for name in REQUIRED_SOURCES:
        pin(pins, root/name)
    # Split audit and all inputs used to produce it are independently byte-bound.
    for path, digest in split['source_pins'].items():
        pin(pins, root/path, digest)
    for name, digest in split['file_sha256'].items():
        pin(pins, root/'data/expanded_validation_v10'/name, digest)
    for name in ('features/manifest.csv', 'features/unlabeled.csv', 'data/raw/submission.csv',
                 'data/expanded_validation_v10/summary.json',
                 'models/hsemotion/enet_b2_7.pt', 'models/hsemotion/enet_b2_7_state.pt',
                 'models/hsemotion/official_weight.json', 'models/hsemotion/conversion_receipt.json'):
        pin(pins, root/name)
    for path in sorted((root/'vendor/hse_runtime/timm').rglob('*.py')):
        pin(pins, path)
    for name in ('METADATA', 'WHEEL'):
        pin(pins, root/'vendor/hse_runtime/timm-0.9.16.dist-info'/name)
    for name, digest in PRESERVED_SUBMISSIONS.items():
        pin(pins, root/name, digest)
    receipt = model_api.source_receipt(root)
    plan = {
        'format': trainer.FORMAT+'_locked', 'state': 'locked', 'classes': h.CLASSES,
        'settings': trainer.SETTINGS, 'source_pins': pins,
        'split_manifest_path': str(root/'data/expanded_validation_v10/manifest.csv'),
        'split_manifest_sha256': h.sha256(root/'data/expanded_validation_v10/manifest.csv'),
        'split_summary_path': str(root/'data/expanded_validation_v10/summary.json'),
        'split_summary_sha256': h.sha256(root/'data/expanded_validation_v10/summary.json'),
        'split_receipt': split, 'input_receipt': split,
        'original_pool_indices': split['original_pool_indices'], 'excluded_indices': [7727,10448],
        'parent_checkpoint': str(root/'models/hsemotion/enet_b2_7.pt'),
        'parent_checkpoint_sha256': model_api.OFFICIAL_SHA,
        'model_source_receipt': receipt, 'model_source_receipt_sha256': h.signature(receipt),
        'grouping_receipt_sha256': h.sha256(root/'data/expanded_validation_v10/summary.json'),
        'old_incumbent_directory': str(root/'ddamfn_v8_results'),
        'incumbent_submission': str(root/'ddamfn_v8_results/submission.csv'),
        'incumbent_submission_sha256': PRESERVED_SUBMISSIONS['ddamfn_v8_results/submission.csv'],
        'candidate_status': 'unranked_against_v8',
        'selection': 'fixed epoch12 EMA, equal original/flip average; epoch0 diagnostic only',
        'old_model_evaluation_allowed': False, 'old_probability_fusion_allowed': False,
        'internal_test_read': False, 'external_answers_used': False,
        'image_content_signature_algorithm': 'SHA256 of concatenated ordered lowercase image SHA256 ASCII hex strings',
    }
    for split_name in ('train','val'):
        records = rows[split_name]
        plan[split_name+'_indices'] = [r['manifest_index'] for r in records]
        plan[split_name+'_rows_sha256'] = h.signature([
            {k:r[k] for k in ('manifest_index','path','label','target')} for r in records])
        plan[split_name+'_image_rows_sha256'] = h.signature([(r['manifest_index'],r['image_sha256']) for r in records])
        plan[split_name+'_image_content_sha256'] = hashlib.sha256(''.join(r['image_sha256'] for r in records).encode('ascii')).hexdigest()
    return plan


def verify_plan(root=ROOT):
    path = Path(root)/'logs'/f'{PREFIX}_plan.json'
    plan = read(path)
    # Check the already frozen hashes before deriving a fresh expected plan.
    for filename, digest in plan['source_pins'].items():
        h.check_hash(filename, digest)
    if plan != build_plan(root):
        raise ValueError('V10 immutable plan differs from current verified inputs')
    trainer.checked_plan(root)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare','verify'), default='verify')
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT:
        raise ValueError('Fixed project root required')
    path = ROOT/'logs'/f'{PREFIX}_plan.json'
    if args.stage == 'prepare' and not path.exists():
        h.write_json(path, build_plan(ROOT))
    plan = verify_plan(ROOT)
    print(json.dumps({'state':'verified','plan_sha256':h.sha256(path),
                     'train':len(plan['train_indices']),'val':len(plan['val_indices']),
                     'source_files':len(plan['source_pins'])}), flush=True)


if __name__ == '__main__':
    main()
