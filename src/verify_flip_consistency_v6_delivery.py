"""Read-only CPU delivery verification; writes only a new verification receipt."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import finalize_flip_consistency_v6 as f
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'lora_flip_consistency_v6_results'


def main():
    plan, recipe = f.verify_locked()
    summary = c.read_json(OUT / 'summary.json')
    assert summary['state'] == 'complete' and summary['classes'] == f.CLASSES
    assert summary['recipe_sha256'] == h.sha256(OUT / 'selected_recipe.json')
    assert summary['plan_sha256'] == h.sha256(f.PLAN)
    for name, digest in summary['file_sha256'].items():
        h.check_hash(f.safe_child(OUT, name), digest)
    print(json.dumps({'stage': 'producer_and_output_hashes_verified',
                      'producer_files': len(recipe['producer_files'])}), flush=True)

    rows, receipt = h.load_trainval(ROOT)
    assert receipt == plan['input_receipt']
    assert [row['manifest_index'] for row in rows['train']] == plan['original_train_indices']
    rows['train'] = [row for row in rows['train'] if row['manifest_index'] not in {7727, 10448}]
    assert len(rows['train']) == 21246 and len(rows['val']) == 2656
    assert h.signature(rows['train']) == plan['filtered_membership_sha256']
    assert [row['manifest_index'] for row in rows['train']] == plan['train_indices']
    assert [row['manifest_index'] for row in rows['val']] == recipe['val_indices'] == plan['val_indices']
    targets = np.asarray([int(row['target']) for row in rows['val']])

    incumbent = c.check_probability(np.load(f.BASELINE / 'selected_val_probabilities.npy', allow_pickle=False), 2656)
    incumbent_metrics = c.metrics(targets, incumbent)
    assert incumbent_metrics == summary['baseline_metrics'] == recipe['baseline_metrics']
    assert incumbent_metrics['correct_count'] == 1915
    old_views = [c.check_probability(np.load(f.PARENT_FOLDER / f'candidates/epoch_05_ema_{view}.npy', allow_pickle=False), 2656)
                 for view in ('original', 'flip')]
    best_count, selected = 1915, None
    records = [{'name': 'previous_best', 'mode': 'incumbent', **incumbent_metrics}]
    compact_arms, paired = {}, {}
    for mode in f.MODES:
        training = c.read_json(f.TRAIN / mode / 'summary.json')
        assert training['state'] == 'complete' and training['epochs_completed'] == 4
        assert len(training['candidates']) == 9
        arm_records = []
        for item in training['candidates']:
            views, single = f.candidate_views(mode, item, targets)
            if item['epoch'] == 0:
                assert all(np.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(views, old_views))
                fused_probability = incumbent.copy()
            else:
                fused_probability = f.fixed_replacement(incumbent, old_views, views)
            fused = c.metrics(targets, fused_probability)
            name = mode + '/' + item['name']
            record = f.metric_record(name, mode, item, single, fused)
            records.append(record)
            arm_records.append(record)
            paired[(mode, item['epoch'], item['weight_source'])] = record
            if fused['correct_count'] > best_count:
                best_count, selected = fused['correct_count'], {**item, 'mode': mode}
        new_records = [item for item in arm_records if item['epoch'] > 0]
        arm_summary = summary['arm_comparison'][mode]
        for key, metric_key, candidates in (
            ('best_single', 'single_correct_count', arm_records),
            ('best_fixed_fusion', 'correct_count', arm_records),
            ('best_new_single', 'single_correct_count', new_records),
            ('best_new_fused', 'correct_count', new_records),
        ):
            best = max(candidates, key=lambda item: item[metric_key])
            assert arm_summary[key]['name'] == best['name']
            assert arm_summary[key]['metrics']['correct_count'] == best[metric_key]
        assert arm_summary['candidates'] == arm_records
        compact_arms[mode] = {
            key: {'name': arm_summary[key]['name'],
                  'correct_count': arm_summary[key]['metrics']['correct_count'],
                  'accuracy': arm_summary[key]['metrics']['accuracy']}
            for key in ('best_single', 'best_fixed_fusion', 'best_new_single', 'best_new_fused')
        }
    assert records == recipe['validation_candidates']
    assert selected == recipe['selected'] and best_count == recipe['selected_metrics']['correct_count']
    assert recipe['selected'] is None and recipe['improved'] is False
    assert summary['selected'] == 'previous_best' and summary['improved'] is False
    assert summary['selected_metrics'] == incumbent_metrics
    expected_pairs = []
    for epoch in range(1, 5):
        for source in ('raw', 'ema'):
            control, treatment = (paired[(mode, epoch, source)] for mode in f.MODES)
            expected_pairs.append({
                'epoch': epoch, 'weight_source': source,
                'consistency_minus_ce_single_correct': treatment['single_correct_count'] - control['single_correct_count'],
                'consistency_minus_ce_fused_correct': treatment['correct_count'] - control['correct_count'],
            })
    assert expected_pairs == summary['arm_comparison']['paired_same_epoch_source']
    with (OUT / 'candidate_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        metrics_csv = list(csv.DictReader(stream))
    assert len(metrics_csv) == len(records) == 19
    for actual, expected in zip(metrics_csv, records):
        for key, text in actual.items():
            value = expected.get(key)
            if value is None:
                assert text == ''
            elif isinstance(value, (int, float)):
                assert math.isclose(float(text), value, rel_tol=1e-12, abs_tol=1e-12)
            else:
                assert text == value
    print(json.dumps({'stage': 'all_19_validation_entries_recomputed', 'best_correct_count': best_count}), flush=True)

    competition_rows = c.unlabeled_rows(ROOT)
    assert c.signature(competition_rows) == f.ROW_SHA
    image_receipt = c.image_receipt(ROOT, competition_rows, verify_inventory=False)
    assert image_receipt['image_content_sha256'] == f.IMAGE_SHA
    probability = c.check_probability(np.load(OUT / 'probabilities.npy', allow_pickle=False), 7065)
    f.verify_submission(competition_rows, probability)
    with (ROOT / 'data/raw/submission.csv').open(encoding='utf-8-sig', newline='') as stream:
        template = list(csv.DictReader(stream))
    with (OUT / 'submission.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['ID', 'Emotion']
        submission = list(reader)
    assert len(submission) == len(template) == 7065
    assert [row['ID'] for row in submission] == [row['ID'] for row in template]
    assert all(row['Emotion'] in {'0', '1', '2', '3', '4', '5'} for row in submission)
    assert np.array_equal(np.asarray([int(row['Emotion']) for row in submission]), probability.argmax(1))
    for filename in ('submission.csv', 'probabilities.npy', 'selected_val_probabilities.npy'):
        assert h.sha256(OUT / filename) == h.sha256(f.BASELINE / filename)
    f.verify_preserved()
    assert summary['new_lora_competition_inference'] is False
    assert summary['internal_test_read'] is False and summary['external_answers_used'] is False
    result = {
        'state': 'passed', 'verifier_sha256': h.sha256(__file__),
        'plan_sha256': h.sha256(f.PLAN), 'recipe_sha256': summary['recipe_sha256'],
        'producer_files_verified': len(recipe['producer_files']),
        'output_file_sha256': summary['file_sha256'], 'candidate_entries_recomputed': len(records),
        'trainval_receipt_verified': receipt, 'effective_train_rows': 21246,
        'competition_image_content_sha256': f.IMAGE_SHA, 'competition_row_order_sha256': f.ROW_SHA,
        'submission_rows': 7065, 'columns': ['ID', 'Emotion'],
        'official_classes': f.CLASSES, 'id_order_verified': True, 'argmax_verified': True,
        'selected': 'previous_best', 'improved': False,
        'baseline_correct_count': 1915, 'selected_correct_count': best_count,
        'arm_comparison': compact_arms, 'paired_same_epoch_source': expected_pairs,
        'preserved_submissions_verified': f.PRESERVED,
        'internal_test_read': False, 'external_answers_used': False,
    }
    h.write_json(OUT / 'delivery_verification.json', result)
    print(json.dumps({key: result[key] for key in ('state', 'producer_files_verified', 'submission_rows', 'selected', 'arm_comparison')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
