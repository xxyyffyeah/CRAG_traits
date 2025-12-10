"""
Small-scale test script for trait-aware evaluation
Uses 10 samples for quick testing
"""

import sys
import os

# Modify the main script to use small sample size
if __name__ == '__main__':
    # Override n_samples in evaluate_with_traits
    import evaluate_with_traits

    # Change to small test size
    evaluate_with_traits.n_samples = 1
    evaluate_with_traits.K_list = [10]  # Fewer K values for testing

    print("\n" + "="*80)
    print("SMALL-SCALE TEST (10 samples, K=[5,10,15])")
    print("="*80 + "\n")

    # Run main
    evaluate_with_traits.main()
