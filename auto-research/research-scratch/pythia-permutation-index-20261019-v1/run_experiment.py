"""Bounded, inference-only Pythia-70M permutation index experiment."""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import platform
import time
import numpy as np
import scipy
from scipy.stats import spearmanr
import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from features import (E8P, Sketch, orbit_hash, permute_mlp, permute_heads, relative,
                      scalar_encode, validate_features, weight_descriptor, weight_marginals)


HERE = pathlib.Path(__file__).resolve().parent
SEEDS = [73, 191, 307]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def state(layer):
    return {k: v.clone() for k, v in layer.state_dict().items()}


def f(layer, x):
    return layer.mlp(layer.post_attention_layernorm(x))


def modify(layer, original, kind, seed, strength, older):
    layer.load_state_dict(original)
    if kind == 'exact_permutation':
        permute_mlp(layer, seed)
    elif kind == 'broken_pairing':
        permute_mlp(layer, seed, coherent=False)
    elif kind == 'noise':
        g = torch.Generator().manual_seed(seed)
        for w in [layer.mlp.dense_h_to_4h.weight, layer.mlp.dense_4h_to_h.weight]:
            v = torch.randn(w.shape, generator=g)
            w.add_(v * (strength * w.norm() / v.norm()))
    elif kind == 'checkpoint64000':
        layer.mlp.load_state_dict(older.mlp.state_dict())
        layer.post_attention_layernorm.load_state_dict(older.post_attention_layernorm.state_dict())


def quantize_all(array, method, e8):
    outputs, codes, scales = [], [], []
    for z in array:
        if method.startswith('e8p'):
            v, code, scale = e8.encode(z, int(method[-1]))
            codes.append(code)
            scales.append(scale)
        else:
            v = scalar_encode(z, int(method[-1]))
        outputs.append(v)
    return torch.stack(outputs), codes, scales


def pairwise(a):
    a = a.double()
    n = a.square().sum(1)
    d = (n[:, None] + n[None, :] - 2 * a @ a.T).clamp_min(0)
    return (d / ((n[:, None] + n[None, :]) / 2).clamp_min(1e-30)).sqrt()


def retrieve(database, queries, labels):
    # Near-zero numerical tolerances only; ties receive fractional credit.
    d = torch.cdist(queries.double(), database.double())
    keys = list(dict.fromkeys(labels))
    representative = [labels.index(k) for k in keys]
    d = d[:, representative]
    scores, ties = [], []
    for i in range(len(queries)):
        hit = torch.where(d[i] <= d[i].min() + 1e-8)[0].tolist()
        target = keys.index(labels[i])
        scores.append((1 / len(hit)) if target in hit else 0.)
        ties.append(len(hit))
    return {'accuracy_with_fractional_ties': float(np.mean(scores)), 'ambiguous_queries': sum(n > 1 for n in ties),
            'database_orbits': len(keys), 'queries': len(queries)}


@torch.inference_mode()
def main(args):
    started = time.perf_counter()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261019)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / 'progress.jsonl').open('w')
    def emit(event, **kw):
        row = {'event': event, 'elapsed_seconds': time.perf_counter() - started, **kw}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + '\n'); log.flush()
    feature_validation = validate_features()
    emit('feature_validation', **feature_validation)
    model = GPTNeoXForCausalLM.from_pretrained(args.cache / '143000', dtype=torch.float32,
                                              attn_implementation='eager', local_files_only=True).eval()
    old = GPTNeoXForCausalLM.from_pretrained(args.cache / '64000', dtype=torch.float32,
                                            attn_implementation='eager', local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.cache / '143000', local_files_only=True)
    passages = json.loads((HERE / 'probes.json').read_text())
    tokens = {s: [[0] + tokenizer.encode(t, add_special_tokens=False)[:127] for t in passages[s]]
              for s in ['calibration', 'holdout']}
    write(args.output / 'tokens.json', tokens)
    params = sum(p.numel() for p in model.parameters())
    pools, locations = {}, {}
    for split in ['calibration', 'holdout']:
        pool, loc = [[] for _ in range(6)], [[] for _ in range(6)]
        for doc, ids in enumerate(tokens[split]):
            hs = model.gpt_neox(torch.tensor([ids]), use_cache=False, output_hidden_states=True).hidden_states
            for l in range(6):
                for pos in range(5, len(ids)):
                    pool[l].append(hs[l][0, pos].clone())
                    loc[l].append([l, doc, pos])
        g = torch.Generator().manual_seed(421 if split == 'calibration' else 422)
        chosen, where = [], []
        for l in range(6):
            take = 22 if l < 2 else 21
            idx = torch.randperm(len(pool[l]), generator=g)[:take]
            chosen.extend(pool[l][int(j)] for j in idx)
            where.extend(loc[l][int(j)] for j in idx)
        pools[split] = torch.stack(chosen)
        locations[split] = where
    write(args.output / 'probe_locations.json', locations)
    sketch = Sketch(pools['calibration'].numel())
    e8 = E8P()
    # Local diagnostic banks are always identical across layers and variants.
    records, yc, yh, query_sketch, wdesc, query_wdesc, marginals, q_marginals = [], [], [], [], [], [], [], []
    timing = {'forward': [], 'orbit_hash': [], 'sketch': [], 'weight_descriptor': []}
    base_ids = []
    for l in range(6):
        layer = copy.deepcopy(model.gpt_neox.layers[l])
        original = state(layer)
        variants = [('original', 0, 0.)]
        for seed in SEEDS:
            variants += [('exact_permutation', seed, 0.), ('broken_pairing', seed, 0.)]
            variants += [('noise', seed, s) for s in [.001, .01, .05]]
        variants += [('checkpoint64000', 0, 0.)]
        base_ids.append(len(records))
        for kind, seed, strength in variants:
            modify(layer, original, kind, seed, strength, old.gpt_neox.layers[l])
            t = time.perf_counter(); h = orbit_hash(layer); timing['orbit_hash'].append(time.perf_counter() - t)
            t = time.perf_counter(); a, b = f(layer, pools['calibration']), f(layer, pools['holdout']); timing['forward'].append((time.perf_counter() - t) / 2)
            t = time.perf_counter(); sk = sketch(a); timing['sketch'].append(time.perf_counter() - t)
            t = time.perf_counter(); wd = weight_descriptor(layer); timing['weight_descriptor'].append(time.perf_counter() - t)
            # Quantiles are unnecessary on every noisy variant: retain them for the decisive marginal-preserving controls.
            md = weight_marginals(layer) if kind in ['original', 'exact_permutation', 'broken_pairing'] else None
            yc.append(a.flatten()); yh.append(b.flatten()); wdesc.append(wd)
            # Each atlas entry also gets an independently permuted query.
            permute_mlp(layer, 99001 + len(records))
            qh = orbit_hash(layer)
            qa = f(layer, pools['calibration'])
            qs = sketch(qa)
            query_sketch.append(qs); query_wdesc.append(weight_descriptor(layer))
            qm = weight_marginals(layer) if md is not None else None
            marginals.append(md); q_marginals.append(qm)
            rec = {'index': len(records), 'layer': l, 'kind': kind, 'seed': seed, 'strength': strength,
                   'orbit_hash': h, 'query_orbit_hash_matches': qh == h, 'query_output_relative_error': relative(qa, a)}
            records.append(rec)
        emit('layer_complete', layer=l, atlas_entries=len(records))
    yc, yh = torch.stack(yc), torch.stack(yh)
    raw = torch.stack([sketch(v) for v in yc])
    qs = torch.stack(query_sketch)
    wd, qwd = torch.stack(wdesc), torch.stack(query_wdesc)
    labels = [r['orbit_hash'] for r in records]
    methods, descriptors, query_descriptors = [], {}, {}
    for dim in [64, 256, 1024]:
        d = raw[:, :dim]; q = qs[:, :dim]
        name = f'fp32_{dim}'
        descriptors[name], query_descriptors[name] = d, q
        methods.append({'method': name, 'bytes_per_entry': dim * 4, **retrieve(d, q, labels)})
        for coding in ['e8p2', 'e8p4', 'scalar2', 'scalar4']:
            t = time.perf_counter()
            dc, codes, scales = quantize_all(d, coding, e8)
            qc, qcodes, qscales = quantize_all(q, coding, e8)
            elapsed = time.perf_counter() - t
            name = f'{coding}_{dim}'
            descriptors[name], query_descriptors[name] = dc, qc
            payload_bytes = dim * int(coding[-1]) // 8 + (8 if coding == 'e8p4' else 4)
            method = {'method': name, 'bytes_per_entry': payload_bytes, 'encoding_seconds_per_entry': elapsed / (2 * len(records)),
                      **retrieve(dc, qc, labels)}
            if codes:
                basecodes = [c.tobytes() for c in codes]
                method['unique_code_only_keys'] = len(set(basecodes))
                method['permuted_code_key_match_rate'] = np.mean([np.array_equal(c, q) for c, q in zip(codes, qcodes)]).item()
                if dim == 1024:
                    np.savez_compressed(args.output / f'{coding}_codes.npz', codes=np.stack(codes), scales=np.array(scales, dtype=np.float32))
            methods.append(method)
        emit('encoding_complete', dimensions=dim)
    methods.append({'method': 'raw_weight_hadamard_1024', 'bytes_per_entry': 4096, **retrieve(wd, qwd, labels)})
    # Invariance/sensitivity diagnostics compare each variant with its final-checkpoint layer.
    for i, rec in enumerate(records):
        j = base_ids[rec['layer']]
        rec['same_orbit_as_final_layer'] = labels[i] == labels[j]
        rec['calibration_response_relative_change'] = relative(yc[i], yc[j])
        rec['holdout_response_relative_change'] = relative(yh[i], yh[j])
        rec['raw_weight_sketch_relative_change'] = relative(wd[i], wd[j])
        rec['weight_marginal_relative_change'] = relative(marginals[i], marginals[j]) if marginals[i] is not None else None
        rec['descriptor_distances'] = {name: relative(a[i], a[j]) for name, a in descriptors.items()}
    comparisons = []
    for name in descriptors:
        for subset, kinds in [('all_edits', ['broken_pairing', 'noise', 'checkpoint64000']), ('nearby_edits', ['noise', 'checkpoint64000']), ('noise_only', ['noise'])]:
            rows = [r for r in records if r['kind'] in kinds]
            target = np.array([r['holdout_response_relative_change'] for r in rows])
            pred = np.array([r['descriptor_distances'][name] for r in rows])
            comparisons.append({'method': name, 'subset': subset, 'count': len(rows),
                                'spearman': float(spearmanr(pred, target).statistic),
                                'median_absolute_relative_distance_error': float(np.median(np.abs(pred-target) / target))})
    # Holdout pairwise distances assess the atlas beyond layer-origin retrieval.
    cal_pair, hold_pair = pairwise(yc), pairwise(yh)
    tri = torch.triu_indices(len(records), len(records), offset=1)
    cross = []
    mask = np.array([labels[int(i)] != labels[int(j)] for i, j in tri.T])
    for name, a in descriptors.items():
        p = pairwise(a)[tri[0], tri[1]].numpy()[mask]
        target = hold_pair[tri[0], tri[1]].numpy()[mask]
        cross.append({'method': name, 'pair_count': len(target), 'spearman': float(spearmanr(p, target).statistic)})
    np.savez_compressed(args.output / 'atlas_descriptors.npz', raw=raw.numpy(), queries=qs.numpy(),
                        weight=wd.numpy(), weight_queries=qwd.numpy(), holdout_pairwise=hold_pair.numpy(),
                        calibration_pairwise=cal_pair.numpy(), **{k: v.numpy() for k, v in descriptors.items()})
    write(args.output / 'atlas_records.json', records)
    write(args.output / 'retrieval.json', methods)
    write(args.output / 'distance_prediction.json', {'within_layer': comparisons, 'pairwise': cross})
    emit('atlas_complete', entries=len(records), distinct_orbits=len(set(labels)))

    # End-to-end checks use intact, heldout documents, never the pooled residual bank.
    def model_eval():
        logps, top, targets, nlls = [], [], [], []
        for ids in tokens['holdout']:
            x = torch.tensor([ids])
            logits = model(x, use_cache=False).logits[0, :-1]
            target = x[0, 1:]
            nll = torch.nn.functional.cross_entropy(logits, target, reduction='sum')
            pos = torch.linspace(0, len(target)-1, 8).round().long().unique()
            lp = torch.log_softmax(logits[pos], -1)
            logps.append(lp); top.append(lp.argmax(-1)); targets.append(len(target)); nlls.append(float(nll))
        return torch.cat(logps), torch.cat(top), sum(nlls)/sum(targets), nlls, targets
    baseline = model_eval()
    def evaluate_change(name, layer=-1, seed=0):
        lp, top, nll, nlls, ntokens = model_eval()
        # KL calculated in FP64 to reduce cancellation for near-identical models.
        p = baseline[0].double().exp()
        kl = float((p * (baseline[0].double() - lp.double())).sum(-1).mean())
        return {'name': name, 'layer': layer, 'seed': seed, 'kl_nats': kl,
                'selected_logprob_relative_change': relative(lp, baseline[0]),
                'selected_top1_agreement': float((top == baseline[1]).float().mean()),
                'nll_nats': nll, 'nll_change': nll - baseline[2], 'passage_nll_sums': nlls,
                'passage_target_counts': ntokens, 'kl_positions': len(top)}
    whole = [evaluate_change('duplicate')]
    originals = [state(l) for l in model.gpt_neox.layers]
    for seed in SEEDS:
        for l, layer in enumerate(model.gpt_neox.layers): permute_mlp(layer, seed + l)
        whole.append(evaluate_change('all_mlp_exact', seed=seed))
        for l, layer in enumerate(model.gpt_neox.layers): layer.load_state_dict(originals[l])
        for l, layer in enumerate(model.gpt_neox.layers): permute_heads(layer, seed + l)
        whole.append(evaluate_change('all_heads_exact', seed=seed))
        for l, layer in enumerate(model.gpt_neox.layers): layer.load_state_dict(originals[l])
        emit('whole_model_exact_complete', seed=seed)
    for l, layer in enumerate(model.gpt_neox.layers):
        for kind, strength in [('broken_pairing', 0.), ('noise', .05), ('checkpoint64000', 0.)]:
            modify(layer, originals[l], kind, 73, strength, old.gpt_neox.layers[l])
            whole.append(evaluate_change(kind, layer=l, seed=73))
            layer.load_state_dict(originals[l])
        permute_heads(layer, 73, coherent=False)
        whole.append(evaluate_change('broken_head_pairing', layer=l, seed=73))
        layer.load_state_dict(originals[l])
        emit('whole_model_edits_complete', layer=l)
    write(args.output / 'whole_model.json', {'baseline_nll_nats': baseline[2], 'evaluations': whole})
    exact = [r for r in records if r['kind'] == 'exact_permutation']
    checks = {'all_102_query_hashes_match': all(r['query_orbit_hash_matches'] for r in records),
              'all_18_exact_orbit_ids_match': all(r['same_orbit_as_final_layer'] for r in exact),
              'no_edited_orbits_equal_original': all(not r['same_orbit_as_final_layer'] for r in records if r['kind'] in ['broken_pairing','noise','checkpoint64000']),
              'all_query_output_relative_error_below_1e-5': max(r['query_output_relative_error'] for r in records) < 1e-5,
              'all_exact_model_top1_agreement': all(r['selected_top1_agreement'] == 1 for r in whole if r['name'].endswith('_exact')),
              'exact_model_logprob_relative_error_below_1e-5': max(r['selected_logprob_relative_change'] for r in whole if r['name'].endswith('_exact')) < 1e-5,
              'duplicate_repeatable': whole[0]['nll_change'] == 0 and whole[0]['kl_nats'] == 0}
    write(args.output / 'verification.json', {'features': feature_validation, 'checks': checks})
    write(args.output / 'timing.json', {k: {'median_ms': 1000 * np.median(v), 'p95_ms': 1000 * np.quantile(v, .95), 'count':len(v)} for k,v in timing.items()})
    write(args.output / 'environment.json', {'python': platform.python_version(), 'torch':torch.__version__,
          'transformers':transformers.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__,
          'device':'cpu', 'threads':torch.get_num_threads(), 'available_cpus':os.cpu_count(), 'model_parameters':params,
          'calibration_tokens':sum(map(len,tokens['calibration'])), 'holdout_tokens':sum(map(len,tokens['holdout']))})
    emit('completed', all_checks_pass=all(checks.values()))
    write(args.output / 'TERMINAL.json', {'status': 'COMPLETED' if all(checks.values()) else 'COMPLETED_WITH_FAILED_CHECKS',
          'wall_seconds':time.perf_counter()-started, 'training_updates':0, 'atlas_entries':len(records), 'whole_model_evaluations':len(whole)+1})
    assert all(checks.values()), checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    main(parser.parse_args())
