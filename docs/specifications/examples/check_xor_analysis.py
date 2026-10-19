#!/usr/bin/env python3
"""Check the published neural-XOR fixture, not arbitrary MGL packages.

Uses exact rational arithmetic. Finite-grid agreement is a regression check;
the fixture's all-real equivalence claims have separate algebraic witnesses.
"""
import json
from fractions import Fraction
from itertools import product
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rational(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            "Expected a numeric fixture constant")
    return Fraction(str(value))


def data(source, name):
    return source["tensors"][name]["data"]


def forward(source, x):
    wi, bi = data(source, "W_in"), data(source, "b_in")
    wo, bo = data(source, "W_out"), data(source, "b_out")
    h = [max(Fraction(0), sum((rational(w) * rational(v)
                              for w, v in zip(row, x)), Fraction(0))
                            + rational(b))
         for row, b in zip(wi, bi)]
    y = sum((rational(w) * v for w, v in zip(wo[0], h)), Fraction(0))
    return h, y + rational(bo[0])


def contribution(source, index, x):
    wi, bi, wo = (data(source, name) for name in ("W_in", "b_in", "W_out"))
    z = sum((rational(w) * rational(v) for w, v in zip(wi[index], x)), Fraction(0))
    return rational(wo[0][index]) * max(Fraction(0), z + rational(bi[index]))


def flat(values):
    for value in values:
        if isinstance(value, list):
            yield from flat(value)
        else:
            yield rational(value)


def marginal(source):
    return {
        name: (sorted(flat(tensor["data"])),
               sum((value * value for value in flat(tensor["data"])), Fraction(0)))
        for name, tensor in source["tensors"].items()
    }


def main():
    path = Path(__file__).with_name("xor-analysis.mgl.json")
    package = json.loads(path.read_text(encoding="utf-8"))
    require(package["schema"] == "mgl.analysis/0.2", "Unexpected schema")
    require(package["package"]["status"] == "illustrative", "Fixture status missing")
    families = ("sources", "operations", "contributions", "segments", "claims",
                "evidence", "descriptors", "correspondences", "pass_runs")
    records = [record for family in families for record in package[family]]
    ids = [record["id"] for record in records]
    require(len(ids) == len(set(ids)), "Record IDs must be unique")
    sources = {record["id"]: record for record in package["sources"]}
    claims = {record["id"]: record for record in package["claims"]}
    evidence = {record["id"]: record for record in package["evidence"]}
    contributions = {record["id"]: record for record in package["contributions"]}

    for source in sources.values():
        architecture = source["architecture"]
        require(architecture["activation"] == "relu", "Unsupported fixture activation")
        require(architecture["equation"] ==
                "y = W_out * relu(W_in * x + b_in) + b_out",
                "This checker only checks the declared fixture equation")
        require(architecture["numeric_regime"] == "real-arithmetic",
                "Fixture requires real-arithmetic claims")
        wi, bi, wo, bo = (data(source, name)
                          for name in ("W_in", "b_in", "W_out", "b_out"))
        width = len(wi)
        require(all(len(row) == 2 for row in wi), "Input width mismatch")
        require(len(bi) == width and len(wo) == 1 and len(wo[0]) == width
                and len(bo) == 1, "Incompatible tensor shapes")
        for name, shape, axes in [
            ("W_in", [width, 2], ["hidden", "input"]),
            ("b_in", [width], ["hidden"]),
            ("W_out", [1, width], ["output", "hidden"]),
            ("b_out", [1], ["output"]),
        ]:
            tensor = source["tensors"][name]
            require(tensor["shape"] == shape and tensor["axes"] == axes,
                    f"Incorrect axes or shape: {source['id']} {name}")

    available = {"x"}
    frame = "xor-input"
    require([op["mnemonic"] for op in package["operations"]] ==
            ["AFFINE", "NONLIN", "AFFINE"], "Unexpected lifted fixture operations")
    for operation in package["operations"]:
        require(operation["source"] == "src-xor", "Unexpected operation source")
        require(set(operation["inputs"]).issubset(available), "Unbound source value")
        require(operation["input_frame"] == frame, "Frame mismatch")
        available.update(operation["outputs"])
        frame = operation["output_frame"]
        for tensor in operation.get("bindings", {}).values():
            require(tensor in sources["src-xor"]["tensors"], "Unbound tensor")
    require("y" in available and frame == "xor-output", "Output binding missing")

    for record in contributions.values():
        require(record["source"] in sources, "Missing contribution source")
        if record.get("kind") == "constant-write":
            require(record["binding"]["tensor"] == "b_out", "Missing output bias")
            continue
        index = record["read"]["selector"]["row"]
        require(record["bias"]["selector"]["index"] == index
                and record["write"]["selector"]["column"] == index,
                "Broken read/bias/write selectors")
        require(0 <= index < len(data(sources[record["source"]], "W_in")),
                "Contribution index out of bounds")

    for claim in claims.values():
        require(claim.get("domain") and claim.get("limitations") is not None,
                "Claim scope missing")
        require(all(item in evidence for item in claim["evidence"]),
                "Missing claim evidence")
    for segment in package["segments"]:
        require(all(item in contributions for item in segment["members"]),
                "Missing segment contribution")
        require(all(item in claims for item in segment["claims"]),
                "Missing segment claim")
    for relation in package["correspondences"]:
        require(relation["source"] in sources and relation["target"] in sources,
                "Missing correspondence endpoint")
        require(relation["claim"] in claims, "Missing correspondence claim")
        if isinstance(relation["mapping"], list):
            for pair in relation["mapping"]:
                require(all(item in contributions for item in pair["from"] + pair["to"]),
                        "Unbound contribution map")

    original = sources["src-xor"]
    permuted = sources["src-permuted"]
    broken = sources["src-broken"]
    wide = sources["src-wide"]
    truth = evidence["ev-truth"]
    require(truth["inputs"] == [[0, 0], [0, 1], [1, 0], [1, 1]],
            "Boolean domain is incomplete")
    require(claims["claim-xor"]["domain"]["inputs"] == truth["inputs"],
            "Semantic domain differs from checked domain")
    actual = [forward(original, x) for x in truth["inputs"]]
    require([h for h, _ in actual] == truth["hidden"], "Hidden truth values differ")
    require([y for _, y in actual] == truth["outputs"] == truth["expected_outputs"],
            "Truth table differs from evidence")
    require(all(y == (x[0] ^ x[1]) for x, (_, y) in zip(truth["inputs"], actual)),
            "Boolean XOR check failed")
    require([forward(broken, x)[1] for x in truth["inputs"]] ==
            evidence["ev-broken"]["altered_outputs"], "Negative-control evidence differs")
    require(claims["claim-broken-equivalence"]["status"] == "refuted",
            "Negative control must refute the proposed equivalence")
    counterexample = evidence["ev-broken"]["counterexample"]
    require(forward(original, counterexample["input"])[1] == counterexample["original"]
            and forward(broken, counterexample["input"])[1] == counterexample["altered"]
            and counterexample["original"] != counterexample["altered"],
            "Stored counterexample does not refute equivalence")
    require(marginal(original) == marginal(broken),
            "Broken pairing must preserve the stated marginal descriptor")

    permutation = evidence["ev-permutation"]["witness"]["hidden_permutation"]
    require(sorted(permutation) == [0, 1], "Invalid hidden permutation")
    require(data(permuted, "W_in") == [data(original, "W_in")[i] for i in permutation]
            and data(permuted, "b_in") == [data(original, "b_in")[i] for i in permutation]
            and data(permuted, "W_out")[0] ==
                [data(original, "W_out")[0][i] for i in permutation],
            "Permutation witness does not match tensors")
    require(data(wide, "W_in") == [data(original, "W_in")[i] for i in [0, 0, 1]]
            and data(wide, "b_in") == [data(original, "b_in")[i] for i in [0, 0, 1]]
            and data(wide, "W_out") == [[0.5, 0.5, -2]],
            "Split witness does not match tensors")

    fixture = package["extensions"]["pinn.fixture"]
    coordinates = fixture["finite_regression_grid"]["coordinates"]
    grid = list(product(coordinates, repeat=2))
    require(len(grid) == fixture["finite_regression_grid"]["expected_points"],
            "Unexpected regression grid")
    for x in grid:
        y = forward(original, x)[1]
        require(y == forward(permuted, x)[1] == forward(wide, x)[1],
                f"Known-equivalent fixture mismatch at {x}")
        require(y == contribution(original, 0, x) + contribution(original, 1, x),
                "Original contribution reconstruction failed")
        require(contribution(original, 0, x) ==
                contribution(wide, 0, x) + contribution(wide, 1, x),
                "Many-to-many contribution map failed")
        require(contribution(original, 1, x) == contribution(wide, 2, x),
                "Unsplit contribution map failed")
    example = fixture["off_domain_example"]
    require(forward(original, example["input"])[1] == example["output"],
            "Off-domain example differs")
    print(json.dumps({
        "status": "passed",
        "fixture": path.name,
        "unique_records": len(ids),
        "boolean_cases": len(truth["inputs"]),
        "rational_grid_points": len(grid),
        "coherent_permutation": "witness and finite regression checked",
        "broken_pairing": "marginal descriptors match; stored counterexample verified",
        "width_change": "split witness and contribution map checked",
        "general_mgl_validation": False,
        "automatic_semantic_discovery": False
    }, indent=2))


if __name__ == "__main__":
    main()
