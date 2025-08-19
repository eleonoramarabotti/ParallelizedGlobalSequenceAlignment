import pytest
import sys
from globalAlignment import *

class MockArgs:
    def __init__(self, shape=(0,0), seq1='', seq2='', gapPenalty=-1, match=1, misMatch=-1):
        self.shape = shape
        self.seq1 = seq1
        self.seq2 = seq2
        self.gapPenalty = gapPenalty
        self.match = match
        self.misMatch = misMatch


# checkSequence
def test_checkSequence_validUpperCase():
    assert checkSequence("ACTG", "first sequence") == "ACTG"

def test_checkSequence_validLowerCase():
    assert checkSequence("actg", "second sequence") == "ACTG"

def test_checkSequence_invalidUpperCase():
    with pytest.raises(NucleotideException) as exc_info:
        checkSequence("AHCTG", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence ('AHCTG'): invalid nucleotide 'H'. Sequences must contain only A, C, T, or G."

def test_checkSequence_invalidLowerCase():
    with pytest.raises(NucleotideException) as exc_info:
        checkSequence("ahctg", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence ('AHCTG'): invalid nucleotide 'H'. Sequences must contain only A, C, T, or G."

def test_checkSequence_emptySequence():
    with pytest.raises(EmptySequenceException) as exc_info:
        checkSequence("", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence: the sequence cannot be empty"

def test_checkSequence_emptyLabel():
    with pytest.raises(EmptyLabelException) as exc_info:
        checkSequence("ACTG", "")
    
    assert str(exc_info.value) == f"Insertion error in ACTG: the label can't be empty"

def test_checkSequence_spaces1():
    assert checkSequence("           actg ", "first sequence") == "ACTG"

def test_checkSequence_spaces2():
    assert checkSequence(" gtag      ", "second sequence") == "GTAG"



# createMatrix
def test_createMatrix_ScoreRe():
    args = MockArgs(shape=(3, 4))
    matrix = createMatrix(args, isDirectionMatrix=False)

    expected = np.array([
        [ 0, -1, -2, -3],
        [-1,  0,  0,  0],
        [-2,  0,  0,  0]
    ], dtype=np.int32)

    assert np.array_equal(matrix, expected)

def test_createMatrix_ScoreSq():
    args = MockArgs(shape=(3, 3))
    matrix = createMatrix(args, isDirectionMatrix=False)

    expected = np.array([
        [ 0, -1, -2],
        [-1,  0,  0],
        [-2,  0,  0]
    ], dtype=np.int32)

    assert np.array_equal(matrix, expected)

def test_createMatrix_DirectionsRe():
    args = MockArgs(shape=(3, 4))
    matrix = createMatrix(args, isDirectionMatrix=True)

    expected = np.array([
        ['', LEFT_DIR, LEFT_DIR, LEFT_DIR],
        [UP_DIR,   '',     '',      ''],
        [UP_DIR,   '',     '',      '']
    ], dtype='S9')

    assert np.array_equal(matrix, expected)

def test_createMatrix_DirectionsSq():
    args = MockArgs(shape=(3, 3))
    matrix = createMatrix(args, isDirectionMatrix=True)

    expected = np.array([
        ['', LEFT_DIR, LEFT_DIR],
        [UP_DIR,   '',     ''],
        [UP_DIR,   '',     '']
    ], dtype='S9')

    assert np.array_equal(matrix, expected)

def test_createMatrix_1x1Score():
    args = MockArgs(shape=(1, 1))
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x1Direction():
    args = MockArgs((1, 1))
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([['']], dtype="S9")
    assert np.array_equal(matrix, expected)

def test_createMatrix_4x1Score():
    args = MockArgs((4, 1))
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0], 
                        [-1],
                        [-2], 
                        [-3]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Score():
    args = MockArgs((1, 4))
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0, -1, -2, -3]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_4x1Direction():
    args = MockArgs((4, 1))
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([[''], 
                        [UP_DIR],
                        [UP_DIR], 
                        [UP_DIR]], dtype='S9')
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Direction():
    args = MockArgs((1, 4))
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([['', LEFT_DIR, LEFT_DIR, LEFT_DIR]], dtype='S9')
    assert np.array_equal(matrix, expected)



# calculateAntiDiagonals
def test_calculateAntidiagonals_2x2():
    args = MockArgs(shape=(2, 2))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(1, 0), (0, 1)],
        [(1, 1)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_2x5():
    args = MockArgs(shape=(2, 5))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(1, 0), (0, 1)],
        [(2, 0), (1, 1)],
        [(3, 0), (2, 1)],
        [(4, 0), (3, 1)],
        [(4, 1)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_1x1():
    args = MockArgs(shape=(1, 1))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)]       
    ]
    assert result == expected

def test_calculateAntidiagonals_1x4():
    args = MockArgs(shape=(1, 4))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(1, 0)],
        [(2, 0)],
        [(3, 0)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_4x1():
    args = MockArgs(shape=(4, 1))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(0, 1)],
        [(0, 2)],
        [(0, 3)]         
    ]
    assert result == expected



# calculateSingleCellScore: this function is covered by tests on fillMatrix as it would be difficult to unit test
# fillMatrix
def test_fillMatrix_2x2_mismatch():
    args = MockArgs(shape=(2, 2), seq1 = "A", seq2 = "T")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    
    resultScore, resultDir = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    
    expectedScore = np.array([
        [ 0, -1],
        [-1, -1]], dtype=np.int32)
       
    expectedDir = np.array([
        ['', LEFT_DIR],       
        [UP_DIR, DIAG_DIR]], dtype="S9")

    assert np.array_equal(resultScore, expectedScore)
    assert np.array_equal(resultDir, expectedDir)

def test_fillMatrix_2x2_match():
    args = MockArgs(shape=(2, 2), seq1 = "A", seq2 = "A")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    
    resultScore, resultDir = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    
    expectedScore = np.array([
        [ 0, -1],
        [-1, 1]], dtype=np.int32)
       
    expectedDir = np.array([
        ['', LEFT_DIR],       
        [UP_DIR, DIAG_DIR]], dtype="S9")

    assert np.array_equal(resultScore, expectedScore)
    assert np.array_equal(resultDir, expectedDir)

def test_fillMatrix_3x3():
    args = MockArgs(shape=(3, 3), seq1 = "AC", seq2 = "TC")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    
    resultScore, resultDir = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    
    expectedScore = np.array([
        [ 0, -1, -2],
        [-1, -1, -2],
        [-2, -2,  0]], dtype=np.int32)
       
    expectedDir = np.array([
        ['', LEFT_DIR, LEFT_DIR],       
        [UP_DIR, DIAG_DIR, DIAG_DIR + LEFT_DIR],
        [UP_DIR, DIAG_DIR + UP_DIR, DIAG_DIR]], dtype="S9")

    assert np.array_equal(resultScore, expectedScore)
    assert np.array_equal(resultDir, expectedDir)

def test_fillMatrix_5x5():
    args = MockArgs(shape=(5, 5), seq1 = "ACTG", seq2 = "TCGG")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    
    resultScore, resultDir = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    
    expectedScore = np.array([
        [ 0, -1, -2, -3, -4],
        [-1, -1, -2, -1, -2],
        [-2, -2,  0, -1, -2],
        [-3, -3, -1, -1,  0],
        [-4, -4, -2, -2,  0]], dtype=np.int32)
       
    expectedDir = np.array([
        ['', LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR],       
        [UP_DIR, DIAG_DIR, DIAG_DIR + LEFT_DIR, DIAG_DIR, LEFT_DIR],
        [UP_DIR, DIAG_DIR + UP_DIR, DIAG_DIR, LEFT_DIR, DIAG_DIR + LEFT_DIR],
        [UP_DIR, DIAG_DIR + UP_DIR, UP_DIR, DIAG_DIR, DIAG_DIR],
        [UP_DIR, DIAG_DIR + UP_DIR, UP_DIR, DIAG_DIR + UP_DIR, DIAG_DIR]], dtype="S9")

    assert np.array_equal(resultScore, expectedScore)
    assert np.array_equal(resultDir, expectedDir)

def test_fillMatrix_3x10():
    args = MockArgs(shape=(3, 10), seq1 = "ACTGAAATG", seq2 = "TA")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    
    resultScore, resultDir = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    
    expectedScore = np.array([
        [ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
        [-1, -1, -2, -1, -2, -3, -4, -5, -6, -7],
        [-2,  0, -1, -2, -2, -1, -2, -3, -4, -5]], dtype=np.int32)
       
    expectedDir = np.array([
        ['', LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR],       
        [UP_DIR, DIAG_DIR, DIAG_DIR + LEFT_DIR, DIAG_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, LEFT_DIR, DIAG_DIR + LEFT_DIR, LEFT_DIR],
        [UP_DIR, DIAG_DIR, LEFT_DIR, UP_DIR + LEFT_DIR, DIAG_DIR, DIAG_DIR, DIAG_DIR + LEFT_DIR, DIAG_DIR + LEFT_DIR, LEFT_DIR, LEFT_DIR]], dtype="S9")

    assert np.array_equal(resultScore, expectedScore)
    assert np.array_equal(resultDir, expectedDir)



# getScore
def test_getScore_1():
    args = MockArgs(shape=(7, 4), seq1 = "ACT", seq2 = "TGGACC")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    score = getScore(args, filledScoreMatrix)

    expectedScore = -2 

    assert score == expectedScore



# traceback
def test_traceback_2x2():
    args = MockArgs(shape=(2, 2), seq1 = "A", seq2 = "A")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('A', 'A')]

    assert alignments == expectedAlignments

def test_traceback_5x5():
    args = MockArgs(shape=(5, 5), seq1 = "ACTG", seq2 = "ACTC")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('ACTG', 'ACTC')]

    assert alignments == expectedAlignments

def test_traceback_2x5():
    args = MockArgs(shape=(2, 5), seq1 = "ACTA", seq2 = "A")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('ACTA', 'A---'), ('ACTA', '---A')]

    assert alignments == expectedAlignments

def test_traceback_5x2():
    args = MockArgs(shape=(5, 2), seq1 = "C", seq2 = "TCAC")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('-C--', 'TCAC'), ('---C', 'TCAC')]

    assert alignments == expectedAlignments

def test_traceback_3x10():
    args = MockArgs(shape=(3, 10), seq1 = "CGTAAACGT", seq2 = "TA")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('CGTAAACGT', '--TA-----'), ('CGTAAACGT', '--T-A----'), ('CGTAAACGT', '--T--A---')]

    assert alignments == expectedAlignments



# printPossibleAlignments
def test_printPossibleAlignments_0(capsys):
    alignments = []

    printPossibleAlignments(alignments)

    expectedOutput = ('')

    out, err = capsys.readouterr()

    assert err == ''
    assert out == expectedOutput

def test_printPossibleAlignments_1(capsys):
    alignments = [("G-ATTACA", "GCAT-GCA")]

    printPossibleAlignments(alignments)

    expectedOutput = """G-ATTACA
| || ·||
GCAT-GCA

"""

    out, err = capsys.readouterr()

    assert err == ''
    assert out == expectedOutput

def test_printPossibleAlignments_2(capsys):
    alignments = [("ACGT", "A-GT"), ("A--T", "AGGT")]

    printPossibleAlignments(alignments)
    
    expectedOutput = """ACGT
| ||
A-GT

A--T
|  |
AGGT

"""

    out, err = capsys.readouterr()

    assert err == ''
    assert out == expectedOutput

def test_printPossibleAlignments_3(capsys):
    alignments = [
    ("G-ATTACA", "GCATG-CA"),
    ("GA-TTACA", "G-CATGCA"),
    ("GATT-ACA", "GCA-TGCA"),
    ]

    printPossibleAlignments(alignments)

    expectedOutput = """G-ATTACA
| ||· ||
GCATG-CA

GA-TTACA
|  ·|·||
G-CATGCA

GATT-ACA
|··  ·||
GCA-TGCA

"""

    out, err = capsys.readouterr()
    assert err == ''
    assert out == expectedOutput



# generateLine
def test_generateLine_1():
    assert generateLine(TOP,    cellsAmount=1) == "┌───┐"
    assert generateLine(MIDDLE, cellsAmount=1) == "├───┤"
    assert generateLine(BOTTOM, cellsAmount=1) == "└───┘"

def test_generateLine_2():
    assert generateLine(TOP,    cellsAmount=2) == "┌───┬───┐"
    assert generateLine(MIDDLE, cellsAmount=2) == "├───┼───┤"
    assert generateLine(BOTTOM, cellsAmount=2) == "└───┴───┘"

def test_generateLine_3():
    assert generateLine(TOP,    cellsAmount=5) == "┌───┬───┬───┬───┬───┐"
    assert generateLine(MIDDLE, cellsAmount=5) == "├───┼───┼───┼───┼───┤"
    assert generateLine(BOTTOM, cellsAmount=5) == "└───┴───┴───┴───┴───┘"



# printDirectionMatrix
def test_printDirectionMatrix(capsys):
    args = MockArgs(shape=(7, 5), seq1 = "ACTG", seq2 = "AAGGTT")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    printDirectionMatrix(filledDirectionMatrix)

    expectedOutput = """┌───┬───┬───┬───┬───┐
│   │   │   │   │   │
│   │←  │←  │←  │←  │
├───┼───┼───┼───┼───┤
│  ↑│↖  │   │   │   │
│   │   │←  │←  │←  │
├───┼───┼───┼───┼───┤
│  ↑│↖ ↑│↖  │↖  │↖  │
│   │   │   │←  │←  │
├───┼───┼───┼───┼───┤
│  ↑│  ↑│↖ ↑│↖  │↖  │
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│  ↑│  ↑│↖ ↑│↖ ↑│↖  │
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│  ↑│  ↑│↖ ↑│↖  │  ↑│
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│  ↑│  ↑│↖ ↑│↖ ↑│↖ ↑│
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
"""

    out, err = capsys.readouterr()
    assert err == ''
    assert out == expectedOutput



# main
def test_main1(capsys):
    sys.argv = ["globalAlignment.py", "ACTTGGA", "AG", "-1", "1", "-1"]
    main()

    out, err = capsys.readouterr()

    assert err == ''
    assert "First sequence: ACTTGGA" in out
    assert "Second sequence: AG" in out
    assert """[[ 0 -1 -2 -3 -4 -5 -6 -7]
 [-1  1  0 -1 -2 -3 -4 -5]
 [-2  0  0 -1 -2 -1 -2 -3]]""" in out
    assert """┌───┬───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │   │
│   │←  │←  │←  │←  │←  │←  │←  │
├───┼───┼───┼───┼───┼───┼───┼───┤
│  ↑│↖  │   │   │   │   │   │↖  │
│   │   │←  │←  │←  │←  │←  │←  │
├───┼───┼───┼───┼───┼───┼───┼───┤
│  ↑│  ↑│↖  │↖  │↖  │↖  │↖  │   │
│   │   │   │←  │←  │   │←  │←  │
└───┴───┴───┴───┴───┴───┴───┴───┘
""" in out
    assert """ACTTGGA
|   |  
A---G--

ACTTGGA
|    | 
A----G-

""" in out
    assert "Alignment score: -3"
    assert err == ''


def test_main2(capsys):
    sys.argv = ["globalAlignment.py", "ACA", "AGAAG        ", "-1", "1", "-1"]
    main()

    out, err = capsys.readouterr()

    assert err == ''
    assert "First sequence: ACA" in out
    assert "Second sequence: AGAAG" in out
    assert """[[ 0 -1 -2 -3]
 [-1  1  0 -1]
 [-2  0  0 -1]
 [-3 -1 -1  1]
 [-4 -2 -2  0]
 [-5 -3 -3 -1]]""" in out
    assert """┌───┬───┬───┬───┐
│   │   │   │   │
│   │←  │←  │←  │
├───┼───┼───┼───┤
│  ↑│↖  │   │↖  │
│   │   │←  │←  │
├───┼───┼───┼───┤
│  ↑│  ↑│↖  │↖  │
│   │   │   │←  │
├───┼───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│↖  │
│   │   │   │   │
├───┼───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│↖ ↑│
│   │   │   │   │
├───┼───┼───┼───┤
│  ↑│  ↑│↖ ↑│  ↑│
│   │   │   │   │
└───┴───┴───┴───┘
""" in out
    assert """ACA--
|·|  
AGAAG

AC-A-
|· | 
AGAAG

A-CA-
| ·| 
AGAAG

""" in out
    assert "Alignment score: -1"
    assert err == ''


def test_main3(capsys):
    sys.argv = ["globalAlignment.py", "AA", "AACAGAAGTCAA", "-1", "1", "-1"]
    main()

    out, err = capsys.readouterr()

    assert """First sequence: AA
Second sequence: AACAGAAGTCAA
[[  0  -1  -2]
 [ -1   1   0]
 [ -2   0   2]
 [ -3  -1   1]
 [ -4  -2   0]
 [ -5  -3  -1]
 [ -6  -4  -2]
 [ -7  -5  -3]
 [ -8  -6  -4]
 [ -9  -7  -5]
 [-10  -8  -6]
 [-11  -9  -7]
 [-12 -10  -8]]
┌───┬───┬───┐
│   │   │   │
│   │←  │←  │
├───┼───┼───┤
│  ↑│↖  │↖  │
│   │   │←  │
├───┼───┼───┤
│  ↑│↖ ↑│↖  │
│   │   │   │
├───┼───┼───┤
│  ↑│  ↑│  ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│  ↑│  ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│  ↑│  ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│  ↑│  ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│  ↑│  ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│
│   │   │   │
├───┼───┼───┤
│  ↑│↖ ↑│↖ ↑│
│   │   │   │
└───┴───┴───┘
AA----------
||          
AACAGAAGTCAA

A--A--------
|  |        
AACAGAAGTCAA

-A-A--------
 | |        
AACAGAAGTCAA

A----A------
|    |      
AACAGAAGTCAA

-A---A------
 |   |      
AACAGAAGTCAA

---A-A------
   | |      
AACAGAAGTCAA

A-----A-----
|     |     
AACAGAAGTCAA

-A----A-----
 |    |     
AACAGAAGTCAA

---A--A-----
   |  |     
AACAGAAGTCAA

-----AA-----
     ||     
AACAGAAGTCAA

A---------A-
|         | 
AACAGAAGTCAA

-A--------A-
 |        | 
AACAGAAGTCAA

---A------A-
   |      | 
AACAGAAGTCAA

-----A----A-
     |    | 
AACAGAAGTCAA

------A---A-
      |   | 
AACAGAAGTCAA

A----------A
|          |
AACAGAAGTCAA

-A---------A
 |         |
AACAGAAGTCAA

---A-------A
   |       |
AACAGAAGTCAA

-----A-----A
     |     |
AACAGAAGTCAA

------A----A
      |    |
AACAGAAGTCAA

----------AA
          ||
AACAGAAGTCAA

Alignment score: -8
""" == out
    assert err == ''


def test_main4():
    sys.argv = ["globalAlignment.py", "ACCFTA", "AACAGAAGTCAA", "-1", "1", "-1"]

    with pytest.raises(NucleotideException) as exc_info:
        main()
    
    assert str(exc_info.value) == f"Insertion error in the first sequence ('ACCFTA'): invalid nucleotide 'F'. Sequences must contain only A, C, T, or G."

def test_main5():
    sys.argv = ["globalAlignment.py", "", "ACTG", "-1", "1", "-1"]

    with pytest.raises(EmptySequenceException) as exc_info:
        main()
    
    assert str(exc_info.value) == f"Insertion error in the first sequence: the sequence cannot be empty"