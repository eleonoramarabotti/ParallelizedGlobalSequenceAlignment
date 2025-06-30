Parallel Implementation of Global Sequence Alignment

Politecnico di Milano – MSc in Bioinformatics for Computational Genomics
Scientific Programming Course – Python Project
Author: Eleonora Marabotti

Overview
This tool provides a parallel implementation of the Needleman-Wunsch algorithm, a classical method for global sequence alignment. It computes the optimal end-to-end alignment between two DNA sequences based on user-defined scoring parameters.
The implementation accelerates the most computationally intensive part of the algorithm—the scoring matrix computation—by leveraging parallel processing across matrix antidiagonals.

Key Features
    Parallel Computation:
        Anti-diagonals are processed concurrently using Python's multiprocessing
        Each cell in an anti-diagonal computed in separate processes
        Shared memory via Array for efficient matrix updates

    Biological Validation:
        Strict nucleotide checking (A/C/T/G only)
        Empty sequence detection
        Case-insensitive sequence handling

    Visualization:
        Direction matrix with Unicode arrows (←, ↑, ↖)
        Grid-based matrix display using box-drawing characters
        Alignment output with match/mismatch symbols (| and ·)

    Traceback:
        Finds all optimal alignment paths
        Stack-based path reconstruction
        Handles multiple optimal paths simultaneously

Dependencies
    NumPy