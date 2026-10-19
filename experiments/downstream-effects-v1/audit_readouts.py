"""Separate small replay retaining full vocabulary logits for metric auditing.

This audits first-site top/random/matched conditions in the first and last
layers. It does not select conditions, revise thresholds or rerun the study.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

from pinn_mgl.effects import ActivationEdit, ForwardBudget
from pinn_mgl.native import GPTNeoXProbe
from pinn_mgl.inputs import sha256_file


def main(folder, cache):
    if (folder / "READOUT_AUDIT.json").exists() or (folder / "readout_audit.npz").exists():
        raise ValueError("Audit files already exist; use a fresh replay directory")
    package = json.loads((folder / "analysis.mgl.json").read_text())
    for name, artifact in package["protocol"]["checkpoint"].items():
        if sha256_file(cache / name) != artifact["sha256"]:
            raise ValueError("Native audit checkpoint mismatch")
    dataset = ROOT / "results/pythia-permutation-index-20261019-v1/tokens.json"
    if sha256_file(dataset) != package["protocol"]["input_hashes"][str(dataset.relative_to(ROOT))]:
        raise ValueError("Audit token source mismatch")
    tokens = json.loads(dataset.read_text())["holdout"][0]
    first = next(t for t in package["trials"] if t["condition"] == "layer0-top" and t["passage"] == 0)
    position = first["token_position"]
    ids = []
    for layer in [0, 5]:
        pair = next(p for p in package["matched_pairs"] if p["layer"] == layer and p["seed"] == 73)
        ids += [f"layer{layer}-top", f"layer{layer}-random73", pair["top_condition"]]
    ids = list(dict.fromkeys(ids))
    torch.set_num_threads(1)
    model = GPTNeoXForCausalLM.from_pretrained(str(cache), local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").eval()
    budget = ForwardBudget(max_forwards=7)
    arrays, records = {}, []
    with GPTNeoXProbe(model, budget) as observer:
        baseline = observer.run(tokens, [position]).logits[position]
        arrays["reference"] = baseline
        ref_logprob = torch.log_softmax(torch.from_numpy(baseline.copy()).double(), dim=-1)
        for idx, cid in enumerate(ids):
            c = next(c for c in package["conditions"] if c["id"] == cid)
            trial = next(t for t in package["trials"] if t["condition"] == cid and t["passage"] == 0 and t["token_position"] == position)
            altered = observer.run(tokens, [position], ActivationEdit(c["layer"], position, tuple(c["units"]), c["gain"])).logits[position]
            key = f"edited{idx}";arrays[key] = altered
            lp = torch.log_softmax(torch.from_numpy(altered.copy()).double(), dim=-1)
            kl = float((ref_logprob.exp() * (ref_logprob - lp)).sum())
            nll_change = float(ref_logprob[tokens[position + 1]] - lp[tokens[position + 1]])
            measured = trial["observations"][0]["readout"]
            errors = {"kl_absolute_error": abs(kl - measured["raw_kl"]),
                      "nll_change_absolute_error": abs(nll_change - measured["nll_change"])}
            records.append({"trial_id": trial["id"], "condition": cid, "edited_array": key,
                "target_token": tokens[position + 1], "independent_torch_fp64_kl": kl,
                "independent_torch_fp64_nll_change": nll_change, **errors,
                "passed": max(errors.values()) < 1e-10})
    np.savez_compressed(folder / "readout_audit.npz", **arrays)
    audit = {"kind": "separate-native-replay-audit", "scope": "First holdout site; first/last-layer top, seed73 random and matched-top conditions",
             "native_forwards": budget.used, "native_single_position_readouts": budget.readout_evaluations,
             "array_sha256": sha256_file(folder / "readout_audit.npz"), "records": records,
             "threshold_absolute": 1e-10, "passed": all(r["passed"] for r in records)}
    (folder / "READOUT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache/143000")
    args = parser.parse_args()
    main(args.results, args.cache)
