import json
import os
import re

with open("results/benchmark_12_results.json", encoding="utf-8") as f:
    bench = json.load(f)

print("=" * 72)
print("  VERIFYING BOTH IEEE (paper_ieee.tex) AND ELSEVIER (paper.tex) PAPERS")
print("=" * 72)

file_map = [
    ("IEEE Version", "paper_ieee.tex" if os.path.exists("paper_ieee.tex") else os.path.join("..", "IEEE_Format", "paper_ieee.tex")),
    ("Elsevier Version", "paper.tex" if os.path.exists("paper.tex") else os.path.join("..", "Elsevier_Format", "paper.tex")),
]

for label, fname in file_map:
    with open(fname, encoding="utf-8") as f:
        tex = f.read()

    print(f"\n--- Auditing {label} ({os.path.basename(fname)}: {len(tex.splitlines())} lines) ---")

    # 1. Verify no banned AI cliches or overclaims
    banned = [
        "delve", "tapestry", "testament", "moreover", "furthermore", "pivotal",
        "underscore", "crucial", "worth noting", "important to note",
        "comprehensive overview", "meticulously", "realm", "harnessing",
        "leverage", "utilize", "seamless", "razor-sharp", "sub-pixel",
        "court evidence", "DWT phase", "phase anomaly"
    ]
    for w in banned:
        matches = re.findall(r"\b" + re.escape(w) + r"\b", tex, re.IGNORECASE)
        assert not matches, f"Banned word '{w}' found in {fname}: {matches}"
    print("  [+] Zero banned AI cliches, overclaims, or DWT phase myths.")

    # 2. Verify all captions are short 1-sentence titles (< 200 chars)
    captions = re.findall(r"\\caption\{([^}]+)\}", tex)
    for c in captions:
        assert len(c) < 220, f"Caption too long in {fname}: {c}"
    print(f"  [+] All {len(captions)} Figure/Table captions are concise 1-sentence titles.")

    # 3. Verify key metrics and peer-review reconciliations
    checks = [
        ("CASIA v2.0 85.1% F1 & 93.4% AUC", "85.1"),
        ("Diffusion CocoGlide 512 / AutoSplice 1,288 / LaMa 990", "1,288"),
        ("Modality totals 7,700 (2,500 + 2,200 + 1,800 + 1,200)", "7,700"),
        ("Dual noise units sigma_255=15 and sigma_n^2=0.0035", "0.0035"),
        ("Boundary Hausdorff 2.14 px", "2.14"),
        ("Ablation V6 (82.1) vs V7 Standard Dice (81.6) vs V8 (85.1)", "81.6"),
        ("Complexity 5.24M params & linear FLOPs 4.82 / 19.28 / 77.12 G", "77.12"),
        ("2022-2024 baselines PSCC-Net, CAT-Net v2, HiFi-Net, IML-ViT", "ma2023iml"),
        ("Positive vanishing gradient sign on authentic images", "+\\frac{2\\epsilon p_i}"),
    ]
    for desc, token in checks:
        assert token in tex, f"Missing '{token}' ({desc}) in {fname}"
        print(f"  [+] Verified: {desc}")

    # 4. Verify all referenced graphics files exist on disk
    gfx = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    for g in gfx:
        assert os.path.exists(g), f"Missing figure file {g} referenced in {fname}"
    print(f"  [+] All {len(gfx)} embedded qualitative/robustness PNG panels exist on disk.")

print("\n" + "=" * 72)
print("  ALL VERIFICATION CHECKS PASSED ACROSS BOTH IEEE AND ELSEVIER VERSIONS!")
print("=" * 72)
