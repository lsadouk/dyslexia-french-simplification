"""
Inter-Annotator Agreement — LectureFacile Human Evaluation
============================================================
Computes Cohen's Kappa between 3 teachers who evaluated
15 simplified texts on 5 criteria (C1-C5), scale 1-5.

Usage:
    1. Fill in the ratings in the RATINGS section below
    2. Run: python inter_annotator_agreement.py

Input format:
    Each teacher rates 15 texts on 5 criteria (C1-C5), scale 1-5.
    RATINGS[teacher][text_index][criterion] = score

Output:
    - Pairwise Cohen's Kappa between all 3 teacher pairs
    - Average Kappa per criterion
    - Overall average Kappa
    - Fleiss' Kappa across all 3 raters
    - Mean score per criterion per teacher
    - Summary table
"""

import numpy as np
from itertools import combinations

# ══════════════════════════════════════════════════════════════════
# FILL IN RATINGS HERE AFTER TEACHERS RETURN THEIR FORMS
# ══════════════════════════════════════════════════════════════════
# Format: RATINGS[teacher_id][text_index (0-14)][criterion (C1-C5)]
# Scores: 1 to 5
# Example: RATINGS["T1"][0]["C1"] = 4

RATINGS = {
    "Enseignant_1": {
        # text 0 (T174), text 1 (T263), ... text 14 (T231)
        0:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        1:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        2:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        3:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        4:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        5:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        6:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        7:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        8:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        9:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        10: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        11: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        12: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        13: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        14: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
    },
    "Enseignant_2": {
        0:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        1:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        2:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        3:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        4:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        5:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        6:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        7:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        8:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        9:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        10: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        11: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        12: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        13: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        14: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
    },
    "Enseignant_3": {
        0:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        1:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        2:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        3:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        4:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        5:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        6:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        7:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        8:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        9:  {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        10: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        11: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        12: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        13: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
        14: {"C1": None, "C2": None, "C3": None, "C4": None, "C5": None},
    },
}

TEXT_IDS = ["T174","T263","T314","T191","T140","T092","T316","T054","T278","T104","T091","T284","T101","T325","T231"]
CRITERIA  = ["C1","C2","C3","C4","C5"]
CRITERIA_NAMES = {
    "C1": "Longueur des phrases",
    "C2": "Simplicité du vocabulaire",
    "C3": "Découpage syllabique",
    "C4": "Fidélité au texte original",
    "C5": "Utilisabilité en classe",
}
TEACHERS = list(RATINGS.keys())
N_TEXTS  = 15
SCALE    = [1, 2, 3, 4, 5]

# ══════════════════════════════════════════════════════════════════
# COHEN'S KAPPA (pairwise, weighted linear)
# ══════════════════════════════════════════════════════════════════
def cohens_kappa(ratings_a, ratings_b, scale=SCALE):
    """
    Compute weighted Cohen's Kappa (linear weights) between two raters.
    ratings_a, ratings_b: lists of integer scores (same length)
    """
    n = len(ratings_a)
    assert n == len(ratings_b), "Raters must have same number of items"
    assert n > 0, "No ratings provided"

    max_val = max(scale)
    min_val = min(scale)
    k       = len(scale)

    # Observed agreement matrix
    obs = np.zeros((k, k))
    for a, b in zip(ratings_a, ratings_b):
        i = scale.index(a)
        j = scale.index(b)
        obs[i][j] += 1
    obs /= n

    # Marginals
    row_marginal = obs.sum(axis=1)
    col_marginal = obs.sum(axis=0)

    # Expected agreement
    exp = np.outer(row_marginal, col_marginal)

    # Linear weights
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i][j] = 1 - abs(scale[i] - scale[j]) / (max_val - min_val)

    po = (w * obs).sum()
    pe = (w * exp).sum()

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

# ══════════════════════════════════════════════════════════════════
# FLEISS' KAPPA (3 raters)
# ══════════════════════════════════════════════════════════════════
def fleiss_kappa(ratings_matrix, scale=SCALE):
    """
    ratings_matrix: shape (n_items, n_raters)
    Returns Fleiss' Kappa.
    """
    n_items, n_raters = ratings_matrix.shape
    k = len(scale)

    # Build category matrix
    P_ij = np.zeros((n_items, k))
    for i in range(n_items):
        for r in range(n_raters):
            cat = scale.index(ratings_matrix[i, r])
            P_ij[i, cat] += 1
    P_ij /= n_raters

    # P_i = extent of agreement for item i
    P_i = ((P_ij ** 2).sum(axis=1) - 1/n_raters) / (1 - 1/n_raters)
    P_bar = P_i.mean()

    # p_j = proportion of assignments to category j
    p_j = P_ij.mean(axis=0)
    P_e = (p_j ** 2).sum()

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)

# ══════════════════════════════════════════════════════════════════
# KAPPA INTERPRETATION
# ══════════════════════════════════════════════════════════════════
def interpret_kappa(k):
    if k < 0:      return "Poor (chance agreement or worse)"
    elif k < 0.20: return "Slight"
    elif k < 0.40: return "Fair"
    elif k < 0.60: return "Moderate"
    elif k < 0.80: return "Substantial"
    else:          return "Almost perfect"

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    # Check all ratings filled
    missing = []
    for t in TEACHERS:
        for i in range(N_TEXTS):
            for c in CRITERIA:
                if RATINGS[t][i][c] is None:
                    missing.append(f"{t} / Text {i} ({TEXT_IDS[i]}) / {c}")
    if missing:
        print("⚠️  Missing ratings — fill in RATINGS before running:")
        for m in missing[:10]:
            print(f"   {m}")
        if len(missing) > 10:
            print(f"   ... and {len(missing)-10} more")
        return

    print("\n" + "="*65)
    print("  LectureFacile — Inter-Annotator Agreement")
    print("="*65)

    # Per-criterion analysis
    kappa_by_criterion = {}
    print(f"\n{'Criterion':<30} {'E1-E2':>8} {'E1-E3':>8} {'E2-E3':>8} {'Fleiss':>8} {'Interp.'}")
    print("-"*75)

    for c in CRITERIA:
        ratings = {t: [RATINGS[t][i][c] for i in range(N_TEXTS)] for t in TEACHERS}
        pairs   = list(combinations(TEACHERS, 2))
        kappas  = []
        pair_ks = []
        for t1, t2 in pairs:
            k = cohens_kappa(ratings[t1], ratings[t2])
            kappas.append(k)
            pair_ks.append(k)

        mat = np.array([[RATINGS[t][i][c] for t in TEACHERS] for i in range(N_TEXTS)])
        fk  = fleiss_kappa(mat)
        avg = np.mean(kappas)
        kappa_by_criterion[c] = {"pairs": pair_ks, "fleiss": fk, "avg": avg}

        label = f"{c} — {CRITERIA_NAMES[c]}"
        print(f"{label:<30} {pair_ks[0]:>8.3f} {pair_ks[1]:>8.3f} {pair_ks[2]:>8.3f} {fk:>8.3f}  {interpret_kappa(avg)}")

    # Overall
    all_kappas = [kappa_by_criterion[c]["avg"] for c in CRITERIA]
    all_fleiss = [kappa_by_criterion[c]["fleiss"] for c in CRITERIA]
    print("-"*75)
    print(f"{'OVERALL AVERAGE':<30} {'':>8} {'':>8} {'':>8} {np.mean(all_fleiss):>8.3f}  {interpret_kappa(np.mean(all_kappas))}")

    # Mean scores per criterion
    print(f"\n{'Criterion':<30} {'E1 mean':>9} {'E2 mean':>9} {'E3 mean':>9} {'Overall':>9}")
    print("-"*65)
    for c in CRITERIA:
        means = [np.mean([RATINGS[t][i][c] for i in range(N_TEXTS)]) for t in TEACHERS]
        label = f"{c} — {CRITERIA_NAMES[c]}"
        print(f"{label:<30} {means[0]:>9.2f} {means[1]:>9.2f} {means[2]:>9.2f} {np.mean(means):>9.2f}")

    # Final question (willingness to use)
    print("\n" + "="*65)
    print("  Summary for paper")
    print("="*65)
    avg_fleiss = np.mean(all_fleiss)
    print(f"\nFleiss' Kappa (overall): {avg_fleiss:.3f} — {interpret_kappa(avg_fleiss)}")
    print("\nPer-criterion Fleiss' Kappa:")
    for c in CRITERIA:
        fk = kappa_by_criterion[c]["fleiss"]
        print(f"  {c}: {fk:.3f} — {interpret_kappa(fk)}")

    print("\n✅ Results ready for Section 5.6 of the paper.")

if __name__ == "__main__":
    main()
