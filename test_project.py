import pytest
from maraproject import *

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
    assert checkSequence("ACTG", "first sequence") == None

def test_checkSequence_validLowerCase():
    assert checkSequence("actg", "first sequence") == None

def test_checkSequence_invalidUpperCase():
    with pytest.raises(NucleotideException) as exc_info:
        checkSequence("AHCTG", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence ('AHCTG'): invalid nucleotide 'H'. Sequences must contain only A, C, T, or G."

def test_checkSequence_invalidLowerCase():
    with pytest.raises(NucleotideException) as exc_info:
        checkSequence("ahctg", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence ('ahctg'): invalid nucleotide 'h'. Sequences must contain only A, C, T, or G."

def test_checkSequence_emptySequence():
    with pytest.raises(EmptySequenceException) as exc_info:
        checkSequence("", "first sequence")
    
    assert str(exc_info.value) == f"Insertion error in the first sequence (''): the sequence cannot be empty"

def test_checkSequence_emptyLabel():
    assert checkSequence("ACTG", "") == None


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
    args = MockArgs(shape=(2, 2))
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],
        [(1, 0), (0, 1)],       
        [(1, 1)]         
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

    expectedAlignments = [('ACTA', 'A---'), ('ACTA', 'A')]

    assert alignments == expectedAlignments

def test_traceback_5x2():
    args = MockArgs(shape=(5, 2), seq1 = "C", seq2 = "TCAC")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('C--', 'TCAC'), ('C', 'TCAC')]

    assert alignments == expectedAlignments

def test_traceback_3x10():
    args = MockArgs(shape=(3, 10), seq1 = "CGTAAACGT", seq2 = "TA")
    antidiag = calculateAntidiagonals(args)
    scoreMatrix = createMatrix(args, isDirectionMatrix=False)
    directionMatrix = createMatrix(args, isDirectionMatrix=True)
    filledScoreMatrix, filledDirectionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)

    alignments = traceback(filledDirectionMatrix, args)

    expectedAlignments = [('CGTAAACGT', 'TA-----'), ('CGTAAACGT', 'T-A----'), ('CGTAAACGT', 'T--A---')]

    assert alignments == expectedAlignments


# printPossibleAlignments



# printDirectionMatrix
