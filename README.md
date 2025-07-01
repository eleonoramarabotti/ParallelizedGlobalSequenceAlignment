# **Parallel Implementation of Global Sequence Alignment**

Politecnico di Milano – MSc in Bioinformatics for Computational Genomics

Scientific Programming Course – Python Project

Author: Eleonora Marabotti


## General Description
This tool provides a parallel implementation of the Needleman-Wunsch algorithm, a classical method for global sequence alignment. It computes the optimal end-to-end alignment between two DNA sequences based on user-defined scoring parameters (sequences, gap penalty, match score and mismatch score).
The implementation accelerates the most computationally intensive part of the algorithm (the matrix filling computation across antidiagonals) using multiprocessing. The final output includes the two input sequences, the score matrix, a visual representation of the alignment paths within the direction matrix, all possible optimal alignments (pairs of aligned sequences) that yield this score and and the optimal alignment score, representing the highest achievable similarity between the sequences under the given scoring scheme.


## Key Features
### Parallel Computation
- Anti-diagonals are processed concurrently using Python's multiprocessing.
- Each cell in an anti-diagonal computed in separate processes.
- Shared memory via Array for efficient matrix updates.

### Biological Validation
- Strict nucleotide checking (A/C/T/G only).
- Empty sequence detection.
- Case-insensitive sequence handling.

### Visualization
- Direction matrix with Unicode arrows (←, ↑, ↖).
- Grid-based matrix display using box-drawing characters.
- Alignment output with match/mismatch symbols (| and ·).

### Traceback
- Finds all optimal alignment paths.
- Stack-based path reconstruction.
- Handles multiple optimal paths simultaneously.


## Example Command
    python alignment.py --seq1 ACTGAC --seq2 ACCTGA -gp -2 -m 3 -mm -1


## Expected Output
- Prints the two input sequences.
- Displays the score matrix as a 2D numpy array.
- Shows the direction matrix visualized with arrows and a grid.
- Lists all possible optimal alignments, with matches (|), mismatches (·), and gaps (-) highlighted.
- Prints the final alignment score.