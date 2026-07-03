#!/usr/bin/env python3
"""
Main entry point for Assignment 2: SVD-based Handwritten Digit Classification.

Run:
    python -m src.assignment_2.run

Lefki Athanasopoulou
Numerical Optimization and Large Scale Linear Algebra
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.assignment_2.data_utils import dictionary_format, ima2, load_data
from src.assignment_2.svd_classifier import (
    build_train_val_split,
    classify_test_set,
    compute_svd,
    evaluate_accuracy_vs_basis,
    extract_basis,
    extract_basis_per_class,
)
from src.assignment_2.two_stage import classify_test_set_two_stage
from src.assignment_2.tangent_distance import classify_with_tangent_distance
from src.assignment_2.visualization import (
    plot_accuracy_vs_k,
    plot_confusion_matrix,
    plot_singular_values,
    display_misclassified,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Load data
    # ------------------------------------------------------------------
    print("Loading data ...")
    azip_df, dzip_df, testzip_df, dtest_df = load_data()

    # Convert to numpy
    train_images_full = azip_df.to_numpy().T       # (1707, 256)
    train_labels_full = dzip_df.to_numpy().flatten()  # (1707,)
    test_images_np = testzip_df.to_numpy()           # (256, 2007)
    test_labels_np = dtest_df.to_numpy().flatten()   # (2007,)

    # ==================================================================
    # Train / Validation split
    # ==================================================================
    print("\nSplitting data into train / validation (80/20) ...")
    img_tr, img_val, lbl_tr, lbl_val = build_train_val_split(
        train_images_full, train_labels_full, test_size=0.2, seed=42,
    )

    # Organise training data into per-digit dictionary
    train_classes = dictionary_format(
        pd.Series(lbl_tr), pd.DataFrame(img_tr.T),
    )

    # ==================================================================
    # SVD training
    # ==================================================================
    print("Computing SVD for each digit class ...")
    svd_train = compute_svd(train_classes)

    # ==================================================================
    # Baseline classification with k = 10
    # ==================================================================
    print("\n--- Baseline: k = 10 ---")
    basis_10 = extract_basis(svd_train, k=10)
    preds_10, _ = classify_test_set(test_images_np, basis_10)
    acc_10 = accuracy_score(test_labels_np, preds_10) * 100
    print(f"Test accuracy (k=10): {acc_10:.2f}%")

    # ==================================================================
    # Q1 – Tune k on validation set
    # ==================================================================
    print("\n--- Q1: Tuning k on validation set ---")
    k_values = range(5, 21)
    val_images_for_eval = img_val.T  # (256, N_val)
    accuracy_results = evaluate_accuracy_vs_basis(
        lbl_val, val_images_for_eval, svd_train, k_values,
    )

    results_df = pd.DataFrame(
        list(accuracy_results.items()),
        columns=["Number of Singular Vectors (k)", "Classification Accuracy (%)"],
    ).round(2)
    print(results_df.T.to_string())

    plot_accuracy_vs_k(accuracy_results, filename="q1_accuracy_vs_k.png")

    best_k = max(accuracy_results, key=accuracy_results.get)
    print(f"\nBest k value: {best_k} with accuracy {accuracy_results[best_k]:.2f}%")

    final_basis = extract_basis(svd_train, best_k)
    preds_best, _ = classify_test_set(test_images_np, final_basis)
    final_acc = accuracy_score(test_labels_np, preds_best) * 100
    print(f"Final test accuracy (k={best_k}): {final_acc:.2f}%")

    # ==================================================================
    # Q2 – Per-digit analysis
    # ==================================================================
    print("\n--- Q2: Classification report ---")
    report = classification_report(test_labels_np, preds_best, digits=4)
    print(report)

    cm = confusion_matrix(test_labels_np, preds_best, labels=range(10))
    plot_confusion_matrix(cm, title="Confusion Matrix", filename="q2_confusion_matrix.png")

    display_misclassified(testzip_df, test_labels_np, preds_best, ima2, max_samples=10)

    # ==================================================================
    # Q3 – Per-class k tuning
    # ==================================================================
    print("\n--- Q3: Per-class k tuning (digits 1 and 7) ---")
    plot_singular_values(svd_train, best_k, filename="q3_singular_values.png")

    # Try different k for digits 1 and 7
    k_range_custom = [best_k - 5, best_k - 3, best_k - 1]
    best_custom_acc = 0.0
    best_k1, best_k7 = best_k, best_k

    svd_train_full = compute_svd(
        dictionary_format(pd.Series(train_labels_full), pd.DataFrame(train_images_full.T))
    )

    for k1 in k_range_custom:
        for k7 in k_range_custom:
            k_vals = {1: k1, 7: k7}
            basis_custom = extract_basis_per_class(svd_train_full, k_vals, default_k=best_k)
            preds_c, _ = classify_test_set(val_images_for_eval, basis_custom)
            acc_c = accuracy_score(lbl_val, preds_c) * 100
            print(f"  k_1={k1}, k_7={k7} -> val accuracy: {acc_c:.2f}%")
            if acc_c > best_custom_acc:
                best_custom_acc = acc_c
                best_k1, best_k7 = k1, k7

    print(f"\nBest custom k: digit 1 -> {best_k1}, digit 7 -> {best_k7}")
    k_custom = {1: best_k1, 7: best_k7}
    basis_custom_final = extract_basis_per_class(svd_train_full, k_custom, default_k=best_k)
    preds_custom, _ = classify_test_set(test_images_np, basis_custom_final)
    custom_acc = accuracy_score(test_labels_np, preds_custom) * 100
    print(f"Test accuracy (custom k): {custom_acc:.2f}%")
    print(classification_report(test_labels_np, preds_custom, digits=4))

    cm_custom = confusion_matrix(test_labels_np, preds_custom, labels=range(10))
    plot_confusion_matrix(cm_custom, title="Confusion Matrix (Custom k)", filename="q3_confusion_matrix_custom.png")

    # ==================================================================
    # Optional – Two-stage classification
    # ==================================================================
    print("\n--- Optional: Two-stage classification ---")
    per_class_k = {d: best_k for d in range(10)}
    per_class_k[1] = best_k1
    per_class_k[7] = best_k7
    basis_two_stage = extract_basis_per_class(svd_train_full, per_class_k, default_k=best_k)

    preds_ts, _, n_s1 = classify_test_set_two_stage(
        test_images_np, svd_train_full, basis_two_stage,
    )
    acc_ts = accuracy_score(test_labels_np, preds_ts) * 100
    print(f"Test accuracy (two-stage): {acc_ts:.2f}%")
    print(f"Stage 1 sufficient for {n_s1} / {test_images_np.shape[1]} images "
          f"({100 * n_s1 / test_images_np.shape[1]:.1f}%)")
    print(classification_report(test_labels_np, preds_ts, digits=4))

    cm_ts = confusion_matrix(test_labels_np, preds_ts, labels=range(10))
    plot_confusion_matrix(cm_ts, title="Confusion Matrix (Two-Stage)", filename="optional_two_stage_cm.png")

    # ==================================================================
    # Optional – Tangent distance classification
    # ==================================================================
    print("\n--- Optional: Tangent distance classification ---")
    preds_td, _ = classify_with_tangent_distance(
        test_images_np, azip_df.to_numpy(), dzip_df.to_numpy().flatten(), m=20,
    )
    acc_td = accuracy_score(test_labels_np, preds_td) * 100
    print(f"Tangent distance accuracy: {acc_td:.2f}%")
    print(classification_report(test_labels_np, preds_td, digits=4))

    cm_td = confusion_matrix(test_labels_np, preds_td, labels=range(10))
    plot_confusion_matrix(
        cm_td, title="Confusion Matrix (Tangent Distance)",
        filename="optional_tangent_distance_cm.png",
    )

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline (k=10)           : {acc_10:.2f}%")
    print(f"  Best uniform k={best_k}       : {final_acc:.2f}%")
    print(f"  Custom per-class k        : {custom_acc:.2f}%")
    print(f"  Two-stage                 : {acc_ts:.2f}%")
    print(f"  Tangent distance          : {acc_td:.2f}%")
    out_dir = __import__("pathlib").Path("outputs/assignment_2").resolve()
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
