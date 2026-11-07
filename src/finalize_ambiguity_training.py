"""Verify weighted continuation, compare two fixed fusions, and export the winner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

import validate_cosine_fusion as common
import finalize_hse_specialist as previous

ROOT = Path(__file__).resolve().parents[1]
PRESERVED = 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7'
PROTOCOL = {'candidates': ['previous_best', 'weighted_hse_fixed_50_flip'],
            'weights': {'three_branch_baseline': .5, 'hse_original': .25, 'hse_flip': .25},
            'selection': 'strictly more correct validation predictions required; ties retain previous best',
            'validation_unweighted': True, 'additional_tuning': False}


def training_signature(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def same_state(left, right):
    return set(left)==set(right) and all(torch.equal(left[name], right[name]) for name in left)


def trainval_members(root):
    result = {'train': [], 'val': []}
    with (root/'features'/'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row['split'] not in result:
                continue
            result[row['split']].append({'manifest_index': index, 'path': row['path'], 'label': row['label'],
                                         'target': common.CLASSES.index(row['label'])})
    if len(result['train'])!=21248 or len(result['val'])!=2656:
        raise ValueError('Training/validation membership count changed')
    return result


def verified_training(root, pins):
    folder = root/'experiments'/'hse_ambiguity_v1'/'downweighted'
    summary, config = common.read_json(folder/'summary.json'), common.read_json(folder/'config.json')
    if (summary.get('state')!='complete' or summary.get('epochs_completed')!=4 or summary.get('mode')!='downweighted'
        or summary.get('classes')!=common.CLASSES or summary.get('config')!=config
        or summary.get('run_signature')!=training_signature(config) or summary.get('validation_weights_used') is not False):
        raise ValueError('Weighted training is incomplete or uses a different configuration')
    common.add_pin(pins, folder/'summary.json')
    for name, expected in summary['file_sha256'].items():
        common.add_pin(pins, folder/name, expected)
    for name, expected in config['source_sha256'].items():
        common.add_pin(pins, root/'src'/name, expected)
    plan = common.read_json(root/'logs'/'ambiguity_plan.json')
    for name, expected in plan['source_sha256'].items():
        common.add_pin(pins, root/'src'/name, expected)
    for name, expected in plan['input_sha256'].items():
        common.add_pin(pins, root/name, expected)
    common.add_pin(pins, root/'logs'/'ambiguity_plan.json')
    weights_folder = root/'experiments'/'hse_ambiguity_v1'/'weights'
    common.add_pin(pins, weights_folder/'summary.json', config['weight_summary_sha256'])
    weight_summary = common.read_json(weights_folder/'summary.json')
    for path, expected in weight_summary['binding']['source_sha256'].items():
        common.add_pin(pins, Path(path), expected)
    for name, expected in weight_summary['file_sha256'].items():
        common.add_pin(pins, weights_folder/name, expected)
    rows = trainval_members(root)
    inputs = config['input_receipt']
    common.add_pin(pins, root/'features'/'manifest.csv', inputs['manifest_sha256'])
    common.add_pin(pins, root/'data'/'split_audit.json', inputs['split_audit_sha256'])
    common.add_pin(pins, root/'data'/'original_splits'/'image_inventory.csv', inputs['image_inventory_sha256'])
    for split in ['train', 'val']:
        if (config[split+'_indices'] != [row['manifest_index'] for row in rows[split]]
            or inputs[split+'_rows_sha256'] != training_signature(rows[split])):
            raise ValueError('Weighted probability membership/order changed')
    pointer = common.read_json(folder/'last.json')
    if pointer['epoch']!=4 or pointer['run_signature']!=summary['run_signature']:
        raise ValueError('Final weighted checkpoint pointer differs')
    checkpoint_path = folder/pointer['checkpoint']
    if checkpoint_path.resolve() != (folder/'epochs'/'epoch_04.pt').resolve():
        raise ValueError('Unexpected last weighted checkpoint path')
    common.add_pin(pins, checkpoint_path, pointer['sha256'])
    final = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    best = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    if (final['config']!=config or final['epoch']!=4 or best['config']!=config
        or best['classes']!=common.CLASSES or best['model_provenance']!=config['model_provenance']
        or best['best_epoch']!=summary['best_epoch'] or final['best_epoch']!=summary['best_epoch']
        or len(final['history'])!=4 or not same_state(best['model_state'], final['best_model_state'])):
        raise ValueError('Best weights are not associated with the committed training run')
    raw_original = np.load(folder/'val_original.npy', allow_pickle=False)
    raw_flip = np.load(folder/'val_flip.npy', allow_pickle=False)
    if not np.array_equal(raw_original, final['best_original'].numpy()) or not np.array_equal(raw_flip, final['best_flip'].numpy()):
        raise ValueError('Best validation caches differ from their committed checkpoint')
    original = common.check_probability(raw_original, 2656)
    flip = common.check_probability(raw_flip, 2656)
    if summary['best_epoch']==0:
        initial_path = root/'experiments'/'hse_specialist_v1'/'best.pt'
        common.add_pin(pins, initial_path, config['initial_checkpoint_sha256'])
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        if not same_state(initial['model_state'], best['model_state']):
            raise ValueError('Epoch zero does not preserve the original HSE parameters')
    targets = np.array([row['target'] for row in rows['val']])
    score = common.metrics(targets, original)
    for key in ['accuracy', 'macro_f1', 'log_loss']:
        if not np.isclose(score[key], summary['best_validation'][key], atol=1e-12, rtol=0):
            raise ValueError('Unweighted validation score differs from weighted training receipt')
    return summary, original, flip, pins


def select(root):
    output = root/'ambiguity_results'; output.mkdir(exist_ok=True)
    recipe_path = output/'selected_recipe.json'
    if recipe_path.exists():
        recipe = common.read_json(recipe_path)
        if recipe['protocol']!=PROTOCOL or recipe['finalizer_sha256']!=common.sha(__file__):
            raise ValueError('Existing weighted fusion protocol/source changed')
        common.verify_pins(root, recipe['producer_files'])
        return recipe
    _, previous_recipe, _ = previous.verify_locked(root)
    pins = dict(previous_recipe['producer_files'])
    common.add_pin(pins, root/'hse_results'/'selected_recipe.json')
    old_summary = common.read_json(root/'hse_results'/'summary.json')
    if old_summary['state']!='complete' or old_summary['selected']!=previous_recipe['selected']:
        raise ValueError('Previous best export differs from its locked validation recipe')
    for name, expected in old_summary['file_sha256'].items():
        common.add_pin(pins, root/'hse_results'/name, expected)
    common.add_pin(pins, root/'hse_results'/'summary.json')
    common.checked_hash(root/'hse_results'/'submission.csv', PRESERVED)
    rows = common.val_rows(root); targets = np.array([int(row['target']) for row in rows])
    p0 = common.check_probability(np.load(root/'fusion_validation_results'/'selected_val_probabilities.npy', allow_pickle=False), 2656)
    old_original = common.check_probability(np.load(root/'experiments'/'hse_specialist_v1'/'val_original.npy', allow_pickle=False), 2656)
    old_flip = common.check_probability(np.load(root/'experiments'/'hse_specialist_v1'/'val_flip.npy', allow_pickle=False), 2656)
    baseline = common.check_probability(np.load(root/'hse_results'/'selected_val_probabilities.npy', allow_pickle=False), 2656)
    reconstruction = .5*p0+.25*old_original+.25*old_flip
    if not np.allclose(baseline, reconstruction, atol=1e-12, rtol=0):
        raise ValueError('Stored previous validation baseline differs from its pinned producers')
    common.add_pin(pins, root/'hse_results'/'selected_val_probabilities.npy')
    baseline_score = common.metrics(targets, baseline)
    if baseline_score['correct_count']!=1913 or not np.isclose(baseline_score['accuracy'], .7202560240963856, atol=1e-12):
        raise ValueError('Previous best validation baseline changed')
    training, original, flip, pins = verified_training(root, pins)
    weighted_probability = common.check_probability(.5*p0+.25*original+.25*flip)
    weighted_score = common.metrics(targets, weighted_probability)
    improved = weighted_score['correct_count']>baseline_score['correct_count']
    selected = 'weighted_hse_fixed_50_flip' if improved else 'previous_best'
    common.add_pin(pins, Path(__file__))
    images = common.image_receipt(root, rows, verify_inventory=True)
    if images['image_content_sha256']!=training['config']['input_receipt']['val_image_content_sha256']:
        raise ValueError('Validation image bytes changed after training')
    common.add_pin(pins, root/'features'/'unlabeled.csv')
    common.add_pin(pins, root/'data'/'raw'/'submission.csv')
    common.add_pin(pins, root/'fusion_results'/'probabilities.npy')
    recipe = {'state': 'locked', 'classes': common.CLASSES, 'protocol': PROTOCOL,
        'finalizer_sha256': common.sha(__file__), 'selected': selected, 'requires_new_predictions': improved,
        'baseline_metrics': baseline_score, 'weighted_metrics': weighted_score,
        'best_hse_epoch': training['best_epoch'], 'hse_single_metrics': training['best_validation'],
        'initial_hse_single_metrics': training['initial_validation'], 'producer_files': pins,
        'val_row_order_sha256': common.signature(rows), 'val_image_content_sha256': images['image_content_sha256'],
        'preserved_submission_sha256': PRESERVED, 'internal_test_read': False, 'external_answers_used': False,
        'unweighted_continuation_control_run': False, 'validation_caveat': 'Reused development validation; not a Kaggle score. No unweighted continuation control.'}
    common.write_json(recipe_path, recipe)
    np.save(output/'selected_val_probabilities.npy', weighted_probability if improved else baseline, allow_pickle=False)
    with (output/'candidate_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['name', 'accuracy', 'macro_f1', 'log_loss', 'correct_count'])
        writer.writeheader()
        for name, score in [('previous_best',baseline_score), ('weighted_hse_fixed_50_flip',weighted_score)]:
            writer.writerow({'name': name, **{key: score[key] for key in writer.fieldnames if key!='name'}})
    print(json.dumps({'stage': 'select', 'selected': selected, 'best_hse_epoch': training['best_epoch'],
        'previous_accuracy': baseline_score['accuracy'], 'weighted_accuracy': weighted_score['accuracy']}), flush=True)
    return recipe


def export(root, recipe, device):
    output=root/'ambiguity_results'; path=output/'selected_recipe.json'; recipe_sha=common.sha(path)
    common.verify_pins(root, recipe['producer_files'])
    rows=common.unlabeled_rows(root)
    summary_path=output/'summary.json'
    if summary_path.exists():
        summary=common.read_json(summary_path)
        if summary['state']!='complete' or summary['recipe_sha256']!=recipe_sha:
            raise ValueError('Existing weighted export differs from locked selection')
        for name, expected in summary['file_sha256'].items():
            common.checked_hash(output/name, expected)
        common.submission_check(output/'submission.csv',rows,np.load(output/'probabilities.npy',allow_pickle=False))
        return summary
    def lock_check():
        common.checked_hash(path,recipe_sha)
    if recipe['requires_new_predictions']:
        images=common.image_receipt(root, rows, verify_inventory=True)
        checkpoint=root/'experiments'/'hse_ambiguity_v1'/'downweighted'/'best.pt'
        binding={'recipe_sha256': recipe_sha, 'model_sha256': common.sha(checkpoint), 'classes':common.CLASSES,
                 'row_order_sha256':common.signature(rows),'image_content_sha256':images['image_content_sha256']}
        original,flip=common.cached_prediction(output/'prediction_cache',binding,
            lambda: previous.predict_hse_views(root,checkpoint,rows,True,device,16,lock_check),lock_check)
        p0=common.check_probability(np.load(root/'fusion_results'/'probabilities.npy',allow_pickle=False),len(rows))
        probabilities=common.check_probability(.5*p0+.25*original+.25*flip,len(rows))
        np.save(output/'probabilities.npy',probabilities,allow_pickle=False)
        with (output/'submission.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['ID','Emotion'])
            writer.writerows((row['ID'],int(label)) for row,label in zip(rows,probabilities.argmax(1)))
        mode='weighted_HSE_fixed_fusion'
    else:
        old=root/'hse_results'
        probabilities=common.check_probability(np.load(old/'probabilities.npy',allow_pickle=False),len(rows))
        common.submission_check(old/'submission.csv',rows,probabilities)
        for name in ['submission.csv','probabilities.npy']:
            shutil.copyfile(old/name,output/name)
        mode='previous_best_preserved'
    lock_check();common.submission_check(output/'submission.csv',rows,probabilities)
    common.checked_hash(root/'hse_results'/'submission.csv',PRESERVED)
    summary={'state':'complete','classes':common.CLASSES,'submission_rows':len(rows),'export_mode':mode,
        'recipe_sha256':recipe_sha,'selected':recipe['selected'],'best_hse_epoch':recipe['best_hse_epoch'],
        'initial_hse_single_metrics':recipe['initial_hse_single_metrics'],'hse_single_metrics':recipe['hse_single_metrics'],
        'baseline_metrics':recipe['baseline_metrics'],'weighted_metrics':recipe['weighted_metrics'],
        'row_order_sha256':common.signature(rows),'old_submission_preserved':True,
        'new_hse_competition_inference':recipe['requires_new_predictions'],
        'internal_test_read':False,'external_answers_used':False,'unweighted_continuation_control_run':False,
        'file_sha256':{name:common.sha(output/name) for name in ['submission.csv','probabilities.npy','selected_val_probabilities.npy','candidate_metrics.csv']}}
    common.write_json(summary_path,summary)
    print(json.dumps({'stage':'export','state':'complete','mode':mode,'rows':len(rows),'submission_sha256':summary['file_sha256']['submission.csv']}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    root=args.project_root.resolve();export(root,select(root),args.device)


if __name__=='__main__':
    main()
