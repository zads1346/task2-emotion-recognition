"""Lock training-only grouped CV; outer labels are behind the locked OOF gate.

This module never imports an old train/validation loader. The source split is
filtered before any label/path is used, then CV workers read a train-only CSV.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
import hse_model as model_api
import train_expanded_validation_v10 as v10
import train_hse_specialist as h

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'hse_groupcv_v11'
PLAN = ROOT/'logs/hse_groupcv_v11_plan.json'
DATA = ROOT/'data/hse_groupcv_v11'
TRAIN = ROOT/'experiments/hse_groupcv_v11'
PROTOCOL = ROOT/'research/hse_groupcv_v11_protocol.json'
REVIEW = ROOT/'research/hse_groupcv_v11_protocol_review.json'
PROTOCOL_SHA = 'df8b5884ea716c63d27c5a61590edba693351fd1345eee11329c1c609cc23bd1'
REVIEW_SHA = 'eb310aa94a282d2639c97b1fbda986a1cd95eec407f055a4831463f5fc7a72c1'
V10_MANIFEST = ROOT/'data/expanded_validation_v10/manifest.csv'
V10_SUMMARY = ROOT/'data/expanded_validation_v10/summary.json'
V10_PLAN = ROOT/'logs/hse_expanded_validation_v10_plan.json'
V10_MANIFEST_SHA = 'b5a9d3cb15d6ce6f586a34194d669350a7f7d599265c73afdc1237ef02a98a8a'
V10_SUMMARY_SHA = 'dc6e649c49953223564d44a09e4e096f2c571d260536b975e644a735ee63cd04'
V10_PLAN_SHA = '68e4f40b56bddb779a3aeb1fd096d8c639d306335c2c9d597956af7b79aa2136'
CLASSES = ['Angry','Fear','Happy','Sad','Surprise','Neutral']
MODES = ['full','fc_only']
SETTINGS = {**v10.SETTINGS, 'checkpoint_selection':'fixed epoch12 EMA only'}
CV_RULE = 'fc_only only if pooled OOF correct_count strictly exceeds full and fc_only strictly wins at least2 of3 folds; otherwise full'
FIELDS = ['manifest_index','path','label','target','group','image_sha256','fold']
ROW_FIELDS = ('manifest_index','path','label','target')
REQUIRED_SOURCES = (
    'src/prepare_hse_groupcv_v11.py', 'src/train_hse_groupcv_v11.py',
    'src/finalize_hse_groupcv_v11.py', 'src/run_hse_groupcv_v11.py',
    'src/hse_groupcv_v11_worker.py', 'start_hse_groupcv_v11.ps1',
    'run_hse_groupcv_v11_worker.ps1', 'src/train_expanded_validation_v10.py',
    'src/hse_model.py', 'src/train_hse_specialist.py', 'src/train_hse_sam_v7.py',
    'src/validate_cosine_fusion.py', 'src/run_hse_sam_v7.py')
SOURCE_FILES = REQUIRED_SOURCES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fixed_root(root):
    if Path(root).resolve() != ROOT:
        raise ValueError('V11 requires its fixed project root')
    return ROOT


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('V11 artifact escapes project')
    digest = h.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError('Unexpected V11 input checksum: '+str(path))
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError('Conflicting V11 checksum')
    pins[str(path)] = digest


def verify_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('Missing immutable producer pins')
    for filename, digest in pins.items():
        pin({}, filename, digest)


def pure(rows):
    return [{key: row[key] for key in ROW_FIELDS} for row in rows]


def image_content(rows):
    return hashlib.sha256(''.join(row['image_sha256'] for row in rows).encode('ascii')).hexdigest()


def row_contract(rows, prefix):
    return {prefix+'_indices':[row['manifest_index'] for row in rows],
        prefix+'_rows_sha256':h.signature(pure(rows)),
        prefix+'_image_rows_sha256':h.signature([(row['manifest_index'],row['image_sha256']) for row in rows]),
        prefix+'_image_content_sha256':image_content(rows),
        prefix+'_groups':sorted({row['group'] for row in rows}),
        prefix+'_group_rows_sha256':h.signature([(row['manifest_index'],row['group']) for row in rows]),
        prefix+'_class_counts':{label:sum(row['label']==label for row in rows) for label in CLASSES}}


def protocol():
    h.check_hash(PROTOCOL, PROTOCOL_SHA)
    h.check_hash(REVIEW, REVIEW_SHA)
    document, review = read(PROTOCOL), read(REVIEW)
    if (review.get('state') != 'passed' or review.get('protocol_sha256') != PROTOCOL_SHA
        or review.get('unresolved_blockers') != []
        or document['initialization']['classes'] != CLASSES
        or document['initialization']['official_sha256'] != model_api.OFFICIAL_SHA):
        raise ValueError('The exact independently reviewed V11 protocol is required')
    return document


def _validate_row(raw, with_fold=False):
    row = {key:raw[key] for key in FIELDS if key != 'fold' or with_fold}
    for key in ('manifest_index','target','group') + (('fold',) if with_fold else ()):
        row[key] = int(row[key])
    if (row['label'] not in CLASSES or CLASSES.index(row['label']) != row['target']
        or row['manifest_index'] in (7727,10448)
        or not Path(row['path']).is_absolute() or not Path(row['path']).resolve().is_relative_to(ROOT)
        or len(row['image_sha256']) != 64
        or any(x not in '0123456789abcdef' for x in row['image_sha256'])):
        raise ValueError('Invalid V11 row or forbidden exclusion')
    if with_fold and row['fold'] not in (0,1,2):
        raise ValueError('Unexpected V11 fold')
    return row


def _source_train_rows(verify_images=False):
    h.check_hash(V10_MANIFEST, V10_MANIFEST_SHA)
    h.check_hash(V10_SUMMARY, V10_SUMMARY_SHA)
    summary = read(V10_SUMMARY)
    rows=[]
    with V10_MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        for raw in csv.DictReader(stream):
            # Never inspect or use held-out path/label fields in this branch.
            if raw.get('split') != 'train':
                continue
            if raw['group'] != raw['group_id']:
                raise ValueError('Source group aliases disagree')
            rows.append(_validate_row(raw))
    contract = row_contract(rows,'train')
    if (len(rows)!=19120 or contract['train_indices'] != summary['train_indices']
        or contract['train_rows_sha256'] != summary['train_rows_sha256']
        or contract['train_image_rows_sha256'] != summary['train_image_rows_sha256']
        or contract['train_image_content_sha256'] != summary['split_image_content_sha256']['train']):
        raise ValueError('V10 training pool changed')
    if verify_images:
        _verify_images(rows)
    return rows


def _verify_images(rows):
    for index,row in enumerate(rows,1):
        h.check_hash(row['path'],row['image_sha256'])
        if index % 4000 == 0:
            print(json.dumps({'stage':'v11_training_image_verify','checked':index,'total':len(rows)}),flush=True)


def assign_folds(rows):
    if len(rows)!=19120 or len({r['manifest_index'] for r in rows})!=19120:
        raise ValueError('V11 must use exactly19120 unique training members')
    if [r['manifest_index'] for r in rows] != sorted(r['manifest_index'] for r in rows):
        raise ValueError('V11 training order must match original manifest order')
    labels=np.asarray([r['target'] for r in rows]);groups=np.asarray([r['group'] for r in rows])
    assignments=np.full(len(rows),-1,dtype=np.int64)
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261017)
    for fold,(_,held) in enumerate(splitter.split(np.zeros(len(rows)),labels,groups)):
        if np.any(assignments[held] != -1):
            raise ValueError('A training member enters OOF twice')
        assignments[held]=fold
    if np.any(assignments<0):
        raise ValueError('OOF missed a training member')
    assigned=[{**r,'fold':int(fold)} for r,fold in zip(rows,assignments)]
    folds=[]
    for fold in range(3):
        train=[r for r in assigned if r['fold']!=fold]
        held=[r for r in assigned if r['fold']==fold]
        for key in ('manifest_index','path','group','image_sha256'):
            if {r[key] for r in train} & {r[key] for r in held}:
                raise ValueError('V11 cross-fold overlap in '+key)
        if {r['target'] for r in train} != set(range(6)) or {r['target'] for r in held} != set(range(6)):
            raise ValueError('V11 fold misses a class')
        folds.append({'fold':fold,**row_contract(train,'train'),**row_contract(held,'val')})
    return assigned,folds


def _csv_bytes(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def materialize_data(verify_images=True, write=False):
    protocol()
    rows,folds=assign_folds(_source_train_rows(verify_images))
    payload=_csv_bytes(rows)
    summary={'format':FORMAT+'_training_only_data','state':'complete','classes':CLASSES,
        'protocol_sha256':PROTOCOL_SHA,'source_manifest_path':str(V10_MANIFEST),
        'source_manifest_sha256':V10_MANIFEST_SHA,'source_summary_sha256':V10_SUMMARY_SHA,
        'splitter':{'class':'StratifiedGroupKFold','n_splits':3,'shuffle':True,'random_state':20261017},
        'library_versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},
        'manifest_sha256':hashlib.sha256(payload).hexdigest(),'rows':19120,
        **row_contract(rows,'train'),'folds':folds,
        'grouping':'Reuse frozen label-blind V10 near-duplicate connected components; no regrouping',
        'subject_independence_established':False,'outer_labels_used':False,
        'oof_coverage_once':True,'no_group_or_byte_or_path_cross_fold':True,
        'official_classes_only':True,'internal_test_read':False,'GPU_used':False}
    if write:
        DATA.mkdir(parents=True,exist_ok=True)
        path=DATA/'train_manifest.csv';summary_path=DATA/'summary.json'
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError('Refusing to overwrite different V11 training-only manifest')
        if summary_path.exists() and read(summary_path)!=summary:
            raise ValueError('Refusing to overwrite different V11 training-only summary')
        if not path.exists():
            path.write_bytes(payload)
        if not summary_path.exists():
            h.write_json(summary_path,summary)
    return rows,summary


def _data_rows(summary,verify_images=False):
    path=DATA/'train_manifest.csv'
    h.check_hash(path,summary['manifest_sha256'])
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('V11 training-only CSV schema changed')
        rows=[_validate_row(raw,True) for raw in reader]
    expected,folds=assign_folds([{k:v for k,v in row.items() if k!='fold'} for row in rows])
    if (expected != rows or folds != summary['folds'] or summary['rows']!=19120
        or summary['classes']!=CLASSES or summary['outer_labels_used'] is not False
        or any(summary[key]!=value for key,value in row_contract(rows,'train').items())):
        raise ValueError('V11 deterministic grouping/membership receipt differs')
    if verify_images:
        _verify_images(rows)
    return rows


def verify_reuse_contract(plan):
    """Verify previously delivered V10 full training without opening outer rows."""
    document=protocol(); artifacts=document['selection']['full_reuse_artifact_sha256']
    if len(artifacts)!=11:
        raise ValueError('V11 reuse requires all11 independently reviewed V10 artifacts')
    checked={}
    for filename,digest in artifacts.items():
        pin(checked,ROOT/filename,digest)
    h.check_hash(V10_PLAN,V10_PLAN_SHA)
    prior_plan=read(V10_PLAN)
    config=read(ROOT/'experiments/hse_expanded_validation_v10/config.json')
    summary=read(ROOT/'experiments/hse_expanded_validation_v10/summary.json')
    verify_pins(prior_plan['source_pins'])
    expected={**SETTINGS,'checkpoint_selection':'fixed epoch12 EMA; epoch0 diagnostic only'}
    if (config['settings']!=expected or prior_plan['settings']!=expected or v10.SETTINGS!=expected
        or config['source_pins']!=prior_plan['source_pins'] or summary['source_pins']!=prior_plan['source_pins']
        or config['plan_sha256']!=V10_PLAN_SHA or summary['plan_sha256']!=V10_PLAN_SHA
        or summary['state']!='complete' or summary['epochs_completed']!=12
        or config['classes']!=CLASSES or config['train_sample_weights']!='uniform1'
        or config['validation_weighted'] is not False or config['per_epoch_evaluation'] is not False):
        raise ValueError('V10 reuse recipe or completed training differs')
    for key in ('train_indices','train_rows_sha256','train_image_content_sha256',
                'parent_checkpoint_sha256','model_source_receipt','model_source_receipt_sha256'):
        if config[key]!=plan[key]:
            raise ValueError('V10 full-pool reuse mismatch: '+key)
    if plan['parent_checkpoint_sha256']!=model_api.OFFICIAL_SHA:
        raise ValueError('V11 requires fresh official initialization')
    for filename in ('research/hse_expanded_validation_v10_delivery_root.json',
                     'research/hse_expanded_validation_v10_delivery_independent.json'):
        if read(ROOT/filename).get('state')!='passed':
            raise ValueError('V10 reuse delivery receipt not passed')
    import train_hse_groupcv_v11 as trainer
    code_contract=trainer.full_reuse_contract()
    if code_contract.get('state')!='passed' or trainer.SETTINGS!=SETTINGS:
        raise ValueError('V11 full code cannot establish exact V10 recipe equivalence')
    return {'state':'passed','v10_plan_sha256':V10_PLAN_SHA,'artifact_sha256':checked,
        'full_settings_sha256':h.signature(expected),'train_rows_sha256':plan['train_rows_sha256'],
        'train_image_content_sha256':plan['train_image_content_sha256'],
        'model_source_receipt_sha256':plan['model_source_receipt_sha256'],
        'code_contract':code_contract,'outer_rows_read':False,'new_inference_run':False,
        'epoch0_omission':'V10 diagnostic saves/restores all RNG; V11 reuses identical loader/group_step/EMA and training RNG seed; epoch0 omission does not change the training recipe'}


def build_plan(root=ROOT,verify_images=True):
    fixed_root(root);document=protocol()
    summary=read(DATA/'summary.json');rows=_data_rows(summary,verify_images)
    h.check_hash(V10_PLAN,V10_PLAN_SHA);prior=read(V10_PLAN)
    pins={}
    for filename,digest in prior['source_pins'].items():
        pin(pins,filename,digest)
    for name in REQUIRED_SOURCES:
        pin(pins,ROOT/name)
    for path,digest in ((PROTOCOL,PROTOCOL_SHA),(REVIEW,REVIEW_SHA),(V10_PLAN,V10_PLAN_SHA),
                        (V10_MANIFEST,V10_MANIFEST_SHA),(V10_SUMMARY,V10_SUMMARY_SHA)):
        pin(pins,path,digest)
    for path in (DATA/'train_manifest.csv',DATA/'summary.json'):
        pin(pins,path)
    for filename,digest in document['selection']['full_reuse_artifact_sha256'].items():
        pin(pins,ROOT/filename,digest)
    receipt=model_api.source_receipt(ROOT)
    plan={'format':FORMAT+'_locked','state':'locked','classes':CLASSES,'modes':MODES,
        'settings':SETTINGS,'source_pins':pins,'protocol_path':str(PROTOCOL),
        'protocol_sha256':PROTOCOL_SHA,'protocol_review_sha256':REVIEW_SHA,
        'train_manifest_path':str(DATA/'train_manifest.csv'),
        'train_manifest_sha256':h.sha256(DATA/'train_manifest.csv'),
        'data_summary_path':str(DATA/'summary.json'),'data_summary_sha256':h.sha256(DATA/'summary.json'),
        'training_only_manifest':True,'folds':summary['folds'],**row_contract(rows,'train'),
        'parent_checkpoint':str(ROOT/'models/hsemotion/enet_b2_7.pt'),
        'parent_checkpoint_sha256':model_api.OFFICIAL_SHA,'model_source_receipt':receipt,
        'model_source_receipt_sha256':h.signature(receipt),
        'outer_manifest_path':str(V10_MANIFEST),'outer_manifest_sha256':V10_MANIFEST_SHA,
        'outer_summary_path':str(V10_SUMMARY),'outer_summary_sha256':V10_SUMMARY_SHA,
        'val_indices':prior['val_indices'],'val_rows_sha256':prior['val_rows_sha256'],
        'val_image_rows_sha256':prior['val_image_rows_sha256'],
        'val_image_content_sha256':prior['val_image_content_sha256'],
        'cv_selection_path':str(TRAIN/'cv_selection.json'),
        'cv_selection_lock_path':str(TRAIN/'cv_selection_lock.json'),'cv_selection_rule':CV_RULE,
        'baseline_directory':str(ROOT/'hse_expanded_validation_v10_results'),
        'baseline_correct_count':3240,'baseline_samples':4782,
        'baseline_submission_sha256':document['delivery']['preserve_v10_submission_sha256'],
        'competition_image_content_sha256':document['delivery']['competition_image_content_sha256'],
        'competition_row_order_sha256':document['delivery']['competition_id_order_sha256'],
        'old_v8_submission_sha256':document['delivery']['preserve_v8_submission_sha256'],
        'outer_validation_used_for_cv':False,'internal_test_read':False,'external_answers_used':False,
        'old_probability_fusion_allowed':False,'subject_independence_established':False,
        'pretraining_overlap_audited':False,'GPU_used_by_plan_generation':False}
    plan['full_reuse_receipt']=verify_reuse_contract(plan)
    return plan


def verify_plan(root=ROOT,verify_images=False):
    fixed_root(root);plan=read(PLAN)
    verify_pins(plan['source_pins'])
    expected=build_plan(root,verify_images)
    if expected!=plan:
        raise ValueError('V11 locked plan differs from source/data/protocol/reuse bindings')
    return plan


def _outer_permission(plan,rows):
    """Independently reproduce the training-only OOF choice before outer access."""
    selection_path=TRAIN/'cv_selection.json'; lock_path=TRAIN/'cv_selection_lock.json'
    selection,lock=read(selection_path),read(lock_path)
    sha=h.sha256(selection_path)
    for document,kind in ((selection,FORMAT+'_cv_selection'),(lock,FORMAT+'_cv_selection_lock')):
        if (document.get('format')!=kind or document.get('state')!='locked'
            or document.get('selected_mode')!='fc_only' or document.get('plan_sha256')!=h.sha256(PLAN)
            or document.get('protocol_sha256')!=PROTOCOL_SHA or document.get('source_pins')!=plan['source_pins']
            or document.get('finalizer_sha256')!=h.sha256(ROOT/'src/finalize_hse_groupcv_v11.py')):
            raise ValueError('Outer validation requires the locked source-bound fc_only choice')
    if (lock.get('cv_selection_sha256')!=sha or selection.get('selection_rule')!=CV_RULE
        or selection.get('oof_manifest_indices')!=plan['train_indices']):
        raise ValueError('OOF selection lock or order mismatch')
    verify_pins(selection['producer_files'])
    counts={};wins=0
    for mode in MODES:
        name='oof_'+mode+'.npy';path=TRAIN/name
        h.check_hash(path,selection['file_sha256'][name])
        if selection['producer_files'].get(str(path))!=h.sha256(path):
            raise ValueError('OOF array is not bound in selection producer set')
        probabilities=np.load(path,allow_pickle=False)
        if (probabilities.shape!=(19120,6) or not np.isfinite(probabilities).all()
            or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1.,atol=1e-5,rtol=0)):
            raise ValueError('Invalid training-only OOF probability matrix')
        correct=probabilities.argmax(1)==np.asarray([r['target'] for r in rows])
        metric=selection['pooled_oof_metrics'][mode]
        if metric['correct_count']!=int(correct.sum()) or metric['samples']!=19120:
            raise ValueError('OOF pooled count cannot be reproduced')
        foldcounts=[]
        for fold in range(3):
            mask=np.asarray([r['fold']==fold for r in rows]);count=int(correct[mask].sum())
            metric=selection['fold_metrics'][mode][str(fold)]
            if metric['correct_count']!=count or metric['samples']!=int(mask.sum()):
                raise ValueError('OOF fold correct count cannot be reproduced')
            foldcounts.append(count)
        counts[mode]=(int(correct.sum()),foldcounts)
    wins=sum(a>b for a,b in zip(counts['fc_only'][1],counts['full'][1]))
    if not (counts['fc_only'][0]>counts['full'][0] and wins>=2):
        raise ValueError('fc_only did not meet the preregistered OOF selection rule')
    if selection.get('fc_only_fold_wins')!=wins:
        raise ValueError('Locked OOF fold-win count changed')
    return sha


def records_for(plan,fold,root=ROOT,outer=False,verify_images=False):
    fixed_root(root)
    if plan!=read(PLAN):
        raise ValueError('records_for requires the current byte-bound locked plan')
    h.check_hash(DATA/'summary.json',plan['data_summary_sha256'])
    rows=_data_rows(read(DATA/'summary.json'),verify_images)
    if fold is not None:
        if type(fold) is not int or fold not in (0,1,2) or outer:
            raise ValueError('CV accepts fold0/1/2 and may never access outer validation')
        return pure([r for r in rows if r['fold']!=fold]),pure([r for r in rows if r['fold']==fold])
    if not outer:
        return pure(rows),[]
    _outer_permission(plan,rows)
    h.check_hash(V10_MANIFEST,V10_MANIFEST_SHA)
    evaluation=[]
    with V10_MANIFEST.open(encoding='utf-8-sig',newline='') as stream:
        for raw in csv.DictReader(stream):
            if raw.get('split')!='val':
                continue
            evaluation.append(_validate_row(raw))
    expected=row_contract(evaluation,'val')
    if len(evaluation)!=4782 or any(expected[k]!=plan[k] for k in (
        'val_indices','val_rows_sha256','val_image_rows_sha256','val_image_content_sha256')):
        raise ValueError('The single outer confirmation split changed')
    for key in ('manifest_index','path','group','image_sha256'):
        if {r[key] for r in rows} & {r[key] for r in evaluation}:
            raise ValueError('Outer confirmation is not isolated in '+key)
    if verify_images:
        _verify_images(evaluation)
    return pure(rows),pure(evaluation)


def prepare(root=ROOT):
    fixed_root(root)
    if PLAN.exists():
        return verify_plan(root,verify_images=True)
    materialize_data(verify_images=True,write=True)
    plan=build_plan(root,verify_images=False)
    h.write_json(PLAN,plan)
    return verify_plan(root,verify_images=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=('prepare','verify'),default='verify')
    parser.add_argument('--verify-images',action='store_true')
    args=parser.parse_args()
    plan=prepare(args.project_root) if args.stage=='prepare' else verify_plan(args.project_root,args.verify_images)
    print(json.dumps({'state':'verified','plan_sha256':h.sha256(PLAN),'train':len(plan['train_indices']),
        'folds':[{'fold':f['fold'],'train':len(f['train_indices']),'held_out':len(f['val_indices'])} for f in plan['folds']],
        'source_files':len(plan['source_pins']),'outer_rows_read':False,'GPU_used':False}),flush=True)


if __name__=='__main__':
    main()
