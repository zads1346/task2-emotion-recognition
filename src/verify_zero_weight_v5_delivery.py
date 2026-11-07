"""Independent CPU-only delivery checks; never trains or reads internal-test images."""
from pathlib import Path
import csv
import json
import time
import numpy as np
import train_hse_specialist as h
import validate_cosine_fusion as c
import finalize_zero_weight_v5 as f

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'lora_zero_weight_v5_results'

def main():
    summary = c.read_json(OUT/'summary.json')
    recipe = c.read_json(OUT/'selected_recipe.json')
    plan = c.read_json(ROOT/'logs/zero_weight_v5_plan.json')
    assert summary['state'] == 'complete' and summary['classes'] == recipe['classes'] == plan['classes'] == f.CLASSES
    assert summary['submission_rows'] == 7065
    h.check_hash(OUT/'selected_recipe.json', summary['recipe_sha256'])
    h.check_hash(ROOT/'logs/zero_weight_v5_plan.json', recipe['plan_sha256'])
    h.check_hash(ROOT/'src/finalize_zero_weight_v5.py', recipe['finalizer_sha256'])
    for name, digest in summary['file_sha256'].items(): h.check_hash(f.safe_child(OUT, name), digest)
    c.verify_pins(ROOT, recipe['producer_files'])
    f.verify_preserved()
    val = c.val_rows(ROOT)
    assert [int(row['manifest_index']) for row in val] == plan['val_indices'] == recipe['val_indices']
    targets = np.asarray([int(row['target']) for row in val], dtype=np.int64)
    incumbent = c.check_probability(np.load(ROOT/'lora_task_v4_results/selected_val_probabilities.npy', allow_pickle=False), 2656)
    old_views = [c.check_probability(np.load(ROOT/f'experiments/lora_task_v4/candidates/epoch_05_ema_{view}.npy', allow_pickle=False), 2656)
                 for view in ('original', 'flip')]
    baseline = c.metrics(targets, incumbent)
    assert baseline['correct_count'] == 1915 and baseline == summary['baseline_metrics'] == recipe['baseline_metrics']
    assert recipe['selection_order'] == ['incumbent', 'uniform_control', 'weighted']
    assert recipe['fusion_weights'] == {'hse': .5, 'lora': .2, 'cnn': .2, 'mlp': .1}
    best, winner, best_probability = baseline, None, incumbent
    entries = [{'name': 'previous_best', 'mode': 'incumbent', **baseline}]
    trained_best = {}
    for mode in ('uniform_control', 'weighted'):
        folder = ROOT/'experiments/lora_zero_weight_v5'/mode
        candidates = c.read_json(folder/'candidates.json')
        assert len(candidates) == 9
        single_best = fusion_best = None
        for item in candidates:
            views, single = f.candidate_views(mode, item, targets)
            if item['epoch'] == 0:
                assert all(np.allclose(view, old, atol=1e-5, rtol=1e-5) for view, old in zip(views, old_views))
                probability = incumbent.copy()
            else:
                probability = f.fixed_replacement(incumbent, old_views, views)
            metrics = c.metrics(targets, probability)
            name = f"{mode}/{item['name']}"
            entries.append(f.metric_record(name, mode, item, single, metrics))
            if item['epoch'] > 0:
                if single_best is None or single['correct_count'] > single_best['correct_count']:
                    single_best = {'name': name, **{key: single[key] for key in ('correct_count', 'accuracy', 'macro_f1')}}
                if fusion_best is None or metrics['correct_count'] > fusion_best['correct_count']:
                    fusion_best = {'name': name, **{key: metrics[key] for key in ('correct_count', 'accuracy', 'macro_f1')}}
            if metrics['correct_count'] > best['correct_count']:
                best, winner, best_probability = metrics, {**item, 'mode': mode}, probability
        trained_best[mode] = {'best_new_single_excluding_epoch0': single_best,
                              'best_new_fixed_fusion_excluding_epoch0': fusion_best}
    assert entries == recipe['validation_candidates']
    assert winner == recipe['selected'] and best == recipe['selected_metrics'] == summary['selected_metrics']
    assert bool(winner) == recipe['improved'] == summary['improved']
    saved_val = np.load(OUT/'selected_val_probabilities.npy', allow_pickle=False)
    assert np.allclose(saved_val, best_probability, atol=1e-12, rtol=0)
    rows = c.unlabeled_rows(ROOT)
    assert c.signature(rows) == recipe['competition_row_order_sha256'] == summary['competition_row_order_sha256'] == f.ROW_SHA
    image_receipt = c.image_receipt(ROOT, rows, verify_inventory=False)
    assert image_receipt['image_content_sha256'] == recipe['competition_image_content_sha256'] == summary['competition_image_content_sha256'] == f.IMAGE_SHA
    probabilities = c.check_probability(np.load(OUT/'probabilities.npy', allow_pickle=False), 7065)
    f.verify_submission(rows, probabilities)
    with (OUT/'submission.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); assert reader.fieldnames == ['ID', 'Emotion']; output_rows = list(reader)
    assert len(output_rows) == 7065 and all(row['Emotion'] in {'0','1','2','3','4','5'} for row in output_rows)
    assert [row['ID'] for row in output_rows] == [row['ID'] for row in rows]
    assert np.array_equal(np.asarray([int(row['Emotion']) for row in output_rows]), probabilities.argmax(1))
    if winner is None:
        assert h.sha256(OUT/'submission.csv') == f.PRESERVED['lora_task_v4_results/submission.csv']
        assert np.array_equal(np.load(OUT/'probabilities.npy', allow_pickle=False), np.load(ROOT/'lora_task_v4_results/probabilities.npy', allow_pickle=False))
    receipt = {'state': 'passed', 'classes': f.CLASSES, 'submission_rows': 7065,
               'official_ID_order_and_argmax_verified': True, 'competition_content_verified': True,
               'strict_fixed_selection_recomputed': True, 'producer_files_verified': len(recipe['producer_files']),
               'trained_best_excluding_epoch0': trained_best, 'selected': summary['selected'],
               'selected_correct_count': best['correct_count'], 'improved': summary['improved'],
               'internal_test_images_read': 0, 'external_answers_used': False,
               'submission_sha256': h.sha256(OUT/'submission.csv'),
               'summary_sha256': h.sha256(OUT/'summary.json'),
               'verifier_sha256': h.sha256(__file__), 'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    h.write_json(OUT/'delivery_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
