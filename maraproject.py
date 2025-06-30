# dependencies
# numpy

import argparse
from ctypes import c_char, c_int
import numpy as np
from multiprocessing import Process, Array

# will be used later for direction matrix
LEFT_DIR = '←'.encode('utf-8')
UP_DIR   = '↑'.encode('utf-8')
DIAG_DIR = '↖'.encode('utf-8')

type Params = object
"""
Object containing all the parameters needed for the sequence alignment process.

Attributes:
    match (int): score awarded for a matching pair of nucleotides.
    misMatch (int): penalty for a mismatching pair of nucleotides.
    gapPenalty (int): penalty for introducing a gap in the alignment.
    seq1 (str): the first input sequence to be aligned.
    seq2 (str): the second input sequence to be aligned.
    shape (tuple[int, int]): the dimensions of the matrices.
"""

class EmptySequenceException(Exception):
    """Custom exception raised when the inserted sequence is empty."""
    def __init__(self, sequence: str, label: str):
        """Initialize the EmptySequenceException with a detailed error message.

        Args:
            sequence (str): the empty sequence.
            label (str): a label to identify the sequence.
        """
        message = f"Insertion error in the {label} ('{sequence}'): the sequence cannot be empty"
        super().__init__(message)


class NucleotideException(Exception):
    """Custom exception raised when an invalid nucleotide is found in a sequence.

    This exception is used to signal that a sequence contains characters other than
    the allowed nucleotides: A, C, T, or G.

    """
    def __init__(self, character: str, sequence: str, label: str):
        """Initialize the NucleotideException with a detailed error message.

        Args:
            character (str): the invalid nucleotide character found in the sequence.
            sequence (str): the sequence containing the invalid character.
            label (str): a label to identify the sequence.
        """
        message = f"Insertion error in the {label} ('{sequence}'): invalid nucleotide '{character}'. Sequences must contain only A, C, T, or G."
        super().__init__(message)


def checkSequence(sequence: str, label: str) -> None:
    """Checks if the inserted sequence contains acceptable nucleotides.

    Args:
        sequence (str): the sequence you want to check.
        label (str): a label to identify the sequence.

    Raises:
        NucleotideException: insertion error if a sequence contains invalid characters.
    """
    if not sequence:
        raise EmptySequenceException(sequence, label)
    for nucleotide in sequence:
        if nucleotide.upper() not in "ACTG": # so it's possible to write actg without errors
            raise NucleotideException(nucleotide, sequence, label)
            

def createMatrix(args: Params, *, isDirectionMatrix: bool) -> np.ndarray:
    """Creates and initialize a matrix for sequence alignment.

    Args:
        args (Params): an object containing matrix dimensions and alignment parameters.
        isDirectionMatrix (bool): if True, initializes a direction matrix, if False, 
                                    initializes a score matrix.

    Returns:
        np.ndArray: the initialized score matrix or direction matrix
    """
    matrix = np.zeros(args.shape, dtype = "S9" if isDirectionMatrix else np.int32) 
    # seq1 = x = columns, seq2 = y = rows
    
    for x in range(1, args.shape[1]):
        matrix[0][x] = LEFT_DIR if isDirectionMatrix else x * args.gapPenalty
        # first row of the matrix, for each column add the gap penalty from the cell before

    for y in range(1, args.shape[0]):
        matrix[y][0] = UP_DIR if isDirectionMatrix else y * args.gapPenalty
        # for each row, in position 0 (first column), add the gap penalty from the cell above

    return matrix   
 

def calculateAntidiagonals(args: Params) -> list:
    """Computes the list of anti-diagonals for a matrix of a given shape.

    Args:
        args (Params): an object containing matrix dimensions and alignment parameters.

    Returns:
        list: a list of anti-diagonals, where each anti-diagonal is a list of (x, y) tuples.
    """
    rows, columns = args.shape 
    antiDiagonals = [] # # each singleDiagList is inserted in this list with the last 'append'
    for index in range(columns):
        startAtTopDiagonals = [] # each diagonal is inserted in this list with the first 'append'
        x = index
        y = 0
        for _ in range(min(rows, index + 1)): 
            startAtTopDiagonals.append((x, y)) # here because i want to append the first cell too
            x -= 1
            y += 1
        
        antiDiagonals.append(startAtTopDiagonals)

    for j in range(1, rows):
        startAtRightDiagonals = []
        x = columns - 1
        y = j
        for _ in range(min(columns, rows - j)):
            startAtRightDiagonals.append((x, y))
            x -= 1
            y += 1

        antiDiagonals.append(startAtRightDiagonals)
    
    return antiDiagonals


def calculateSingleCellScore(cell: tuple[int,int], args: Params, scoreMatrix: np.ndarray, directionMatrix: np.ndarray) -> None:
    """Computes the score and direction for a single cell in the alignment matrix.

    Args:
        cell (tuple[int,int]): the (x, y) coordinates of the cell to compute.
        args (Params): an object containing matrix dimensions and alignment parameters.
        scoreMatrix (np.ndarray): the scoring matrix.
        directionMatrix (np.ndarray): the direction matrix.
    """
    x, y = cell
    if y == 0 or x == 0: # we have filled them yet
        return

    scoreMatrix = np.frombuffer(scoreMatrix.get_obj(), dtype=np.int32).reshape(args.shape)
    directionMatrix = np.frombuffer(directionMatrix.get_obj(), dtype='S9').reshape(args.shape)

    upScore = scoreMatrix[y-1][x] + args.gapPenalty
    leftScore = scoreMatrix[y][x-1] + args.gapPenalty
    
    # diagonal move
    if args.seq1[x-1] == args.seq2[y-1]:
        diagScore = scoreMatrix[y-1][x-1] + args.match
    else:
        diagScore = scoreMatrix[y-1][x-1] + args.misMatch
    
    scoreMatrix[y][x] = max(upScore, leftScore, diagScore) # pick the higher

    cellDirections = b""

    # you can have multiple directions (same score)
    if scoreMatrix[y][x] == diagScore:
        cellDirections += DIAG_DIR

    if scoreMatrix[y][x] == upScore:
        cellDirections += UP_DIR
 
    if scoreMatrix[y][x] == leftScore:
        cellDirections += LEFT_DIR

    directionMatrix[y][x] = cellDirections


def fillMatrix(antiDiagonals: list, args: Params, scoreMatrix: np.ndarray, directionMatrix: np.ndarray) -> np.ndarray:
    """Fills the score and direction matrices using parallel processing.

    The function updates the given matrices by computing alignment scores cell-by-cell.

    Args:
        antiDiagonals (list): a list of anti-diagonals, where each anti-diagonal is a list 
                                of (row, column) positions to process.
        args (Params): an object containing matrix dimensions and alignment parameters.
        scoreMatrix (np.ndarray): the scoring matrix you want to fill.
        directionMatrix (np.ndarray): the direction matrix you want to fill.

    Returns:
        np.ndarray: the updated scoreMatrix and directionMatrix.
    """
    directionMatrix = Array(c_char, directionMatrix.tobytes())
    scoreMatrix = Array(c_int, scoreMatrix.flatten())

    # parallelization moment
    for diag in antiDiagonals:
        processes = []
        for cell in diag:
            p = Process(target=calculateSingleCellScore, args=(cell, args, scoreMatrix, directionMatrix))
            p.daemon = True
            p.start()
            processes.append(p)
       
        list(map(lambda p: p.join(), processes))

    return np.frombuffer(scoreMatrix.get_obj(), dtype=np.int32).reshape(args.shape), np.frombuffer(directionMatrix.get_obj(), dtype='S9').reshape(args.shape)



def getScore(args: Params, scoreMatrix: np.ndarray) -> int:
    """Returns the score of the alignment score.

    Args:
        args (Params): an object containing matrix dimensions and alignment parameters.
        scoreMatrix (np.ndarray): the filled score matrix.

    Returns:
        int: the alignment score.
    """
    return scoreMatrix[args.shape[0] - 1, args.shape[1] - 1] 



def traceback(directionMatrix: np.ndarray, args: Params) -> list:
    """Reconstructs all optimal alignments by following the traceback paths.

    Starting from the bottom-right cell of the direction matrix, this function explores
    all possible paths back to the top-left, building aligned sequences along the way.
    It returns all optimal alignments according to the directions encoded in the matrix.

    Args:
        directionMatrix (np.ndarray): the direction matrix
        args (Params): an object containing matrix dimensions and alignment parameters.

    Returns:
        list: a list of tuples, each containing a pair of aligned sequences.
    """
    x = args.shape[1] - 1
    y = args.shape[0] - 1

    stack = [(y, x, "", "")] # starting from the last cell 
    possibleAlignments = [] # where all the alignments are saved (as tuples of two strings)

    while stack: # until the stack is not empty (until we have paths)
        y, x, wipSequence1, wipSequence2 = stack.pop() # removes the last element

        if y == 0 or x == 0: # when you touch one of the two edges (top or left) you add all the remaining sequence (last thing you do, only if the path does not end in 0,0)
            finalSequence1 = args.seq1[:x] + wipSequence1 
            finalSequence2 = args.seq2[:y] + wipSequence2
            possibleAlignments.append((finalSequence1, finalSequence2))
            continue
        
        # defining your current position (remember that the length of the matrix is len(seq)+1) -> last cell 
        nuclLeft = args.seq2[y - 1]
        nuclUp = args.seq1[x - 1]

        directions = directionMatrix[y][x]

        if DIAG_DIR in directions:
            newPosition = [y - 1, x - 1] # diagonal move
            alignedSequences = [nuclUp + wipSequence1, nuclLeft + wipSequence2] # you have to add the nucleotides on both sequences 

            stack.append((*newPosition, *alignedSequences)) # '*' because you have to add everything inside these variables

        if LEFT_DIR in directions:
            newPosition = [y, x - 1] # same row, different column
            alignedSequences = [nuclUp + wipSequence1, '-' + wipSequence2]
            # you have to add the nucleotide in the horizontal sequence, in the other you have a gap
        
            stack.append((*newPosition, *alignedSequences)) # '*' because you have to add everything inside these variables

        if UP_DIR in directions:
            newPosition = [y - 1, x] # same column, different row
            alignedSequences = ['-' + wipSequence1, nuclLeft + wipSequence2]
        
            stack.append((*newPosition, *alignedSequences)) # '*' because you have to add everything inside these variables

    return possibleAlignments


def printPossibleAlignments(possibleAlignments: list[tuple[str, str]]) -> None:
    """Prints all possible pairwise alignments between two sequences, highlighting matches, 
    mismatches, and gaps.

    Args:
        possibleAlignments (list[tuple[str, str]]): a list of tuples, each containing two equal-length strings `seq1` and `seq2`
                                                    (with '-' characters representing gaps).
    """
    for seq1, seq2 in possibleAlignments:
        matchLine = ""
        for nuc1, nuc2 in zip(seq1, seq2):
            if nuc1 == nuc2: # it's never possible for both sequences to have a gap at the same position
                matchLine += '|'
            elif nuc1 == '-' or nuc2 == '-':
                matchLine += ' '
            else:
                matchLine += '·'
        
        print(seq1, matchLine, seq2, sep='\n', end='\n'*2)


# these 'signs' are taken from the GitHub repository for which I'm a contributor (https://github.com/M1keCodingProjects/PyChess)
TOP    = "┌┬┐"
MIDDLE = "├┼┤"
BOTTOM = "└┴┘"
WALL   = '─'

def printDirectionsMatrix(directionMatrix: np.ndarray) -> None:
    """Formats and prints the direction matrix.

    Args:
        directionMatrix (np.ndarray): the filled direction matrix.
    """
    topSeparator = TOP[0] + (WALL * 3 + TOP[1]) * (directionMatrix.shape[1] - 1) + WALL * 3 + TOP[-1]
    middleSeparator = MIDDLE[0] + (WALL * 3 + MIDDLE[1]) * (directionMatrix.shape[1] - 1) + WALL * 3 + MIDDLE[-1]
    bottomSeparator = BOTTOM[0] + (WALL * 3 + BOTTOM[1]) * (directionMatrix.shape[1] - 1) + WALL * 3 + BOTTOM[-1]
    print(topSeparator)
    for y in range(0, 2 * directionMatrix.shape[0]):
        isFirstLine = y % 2 == 0
        rowBuff     = "│"
        for x in range(directionMatrix.shape[1]):
            if isFirstLine:
                rowBuff += DIAG_DIR.decode() + ' '  if DIAG_DIR in directionMatrix[y // 2, x] else '  '
                rowBuff += UP_DIR.decode()          if UP_DIR   in directionMatrix[y // 2, x] else ' '
            else:
                rowBuff += LEFT_DIR.decode() + '  ' if LEFT_DIR in directionMatrix[y // 2, x] else '   '
        
            rowBuff += '│'

        print((middleSeparator + '\n') * isFirstLine * (y != 0) + rowBuff)
    print(bottomSeparator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Scientific Programming Project", description = "Takes two sequences as input for the allignement")
    parser.add_argument("--seq1", type=str, help="Write here your first sequence")
    parser.add_argument("--seq2", type=str, help="Write here your second sequence")
    parser.add_argument("-gp", "--gapPenalty", type=int, help="Write the negative gap penalty you want to apply")
    parser.add_argument("-m", "--match", type=int, help="Write the match score you want to apply")
    parser.add_argument("-mm", "--misMatch", type=int, help="Write the mismatch score you want to apply")

    args = parser.parse_args()
    args.shape = (len(args.seq2) + 1, len(args.seq1) + 1)
    args: Params

    args.seq1 = args.seq1.strip()
    args.seq2 = args.seq2.strip()

    checkSequence(args.seq1, label = "first sequence")
    checkSequence(args.seq2, label = "second sequence")
    scoreMatrix = createMatrix(args, isDirectionMatrix = False)
    directionMatrix = createMatrix(args, isDirectionMatrix = True)
    antidiag = calculateAntidiagonals(args)
    scoreMatrix, directionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    score = getScore(args, scoreMatrix)
    possibleAlignments = traceback(directionMatrix, args)

    
    print("First sequence:", args.seq1)
    print("Second sequence:", args.seq2)
    print(scoreMatrix)
    printDirectionsMatrix(directionMatrix)
    printPossibleAlignments(possibleAlignments)
    print("Alignment score:", score)
    