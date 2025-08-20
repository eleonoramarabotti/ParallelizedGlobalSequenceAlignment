# **Parallel Implementation of Global Sequence Alignment**

Politecnico di Milano – MSc in Bioinformatics for Computational Genomics

Scientific Programming Course – Python Project

Author: Eleonora Marabotti


## General Description
This tool provides a parallel implementation of the Needleman-Wunsch algorithm, a classical method for global sequence alignment. It computes the optimal end-to-end alignments between two DNA sequences based on user-defined scoring parameters (gap penalty, match score and mismatch score).
The use of parallel programming reduced the runtime by accelerating the most computationally intensive part of the algorithm (filling the score and direction matrices). The final output includes all the possible optimal alignments and the optimal alignment score.


## Key Features
### Biological Validation
- Strict nucleotide checking (A/C/T/G only).
- Empty sequence detection.
- Case-insensitive sequence handling.

### Parallel Computation
- The score and direction matrices are numpy arrays and are filled in by anti-diagonals from the top-left corner.
- Each cell in an anti-diagonal is processed in parallel using Python's Multiprocessing module.

### Traceback
- Finds all optimal alignment paths.
- Stack-based path reconstruction.


## Example Command
```shell
python3 globalAlignment.py ACTGAC ACCTGA -gp -1 -m 1 -mm -1
```


## Expected Output
- Prints the two input sequences.
- Displays the score matrix.
- Shows the direction matrix as a table rendered with box-drawing characters, 
while the directions are represented by Unicode arrows (←, ↑, ↖).
- Lists all possible optimal alignments, with matches (|), mismatches (·), and gaps (-) highlighted.
- Prints the final alignment score.


## Requirements
### Mandatory
- Python 3.13.5 or newer
- Numpy

### Pre-installed
- Argparse
- Ctypes
- Multiprocessing

### Test
- Pytest