import pytest
from maraproject import *

class MockArgs:
    def __init__(self, shape, gapPenalty, match, misMatch):
        self.shape = shape
        self.gapPenalty = gapPenalty
        self.match = match
        self.mismatch = misMatch


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
    args = MockArgs(shape=(3, 4), gapPenalty=-1, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=False)

    expected = np.array([
        [ 0, -1, -2, -3],
        [-1,  0,  0,  0],
        [-2,  0,  0,  0]
    ], dtype=np.int32)

    assert np.array_equal(matrix, expected)

def test_createMatrix_ScoreSq():
    args = MockArgs(shape=(3, 3), gapPenalty=-1, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=False)

    expected = np.array([
        [ 0, -1, -2],
        [-1,  0,  0],
        [-2,  0,  0]
    ], dtype=np.int32)

    assert np.array_equal(matrix, expected)

def test_createMatrix_DirectionsRe():
    args = MockArgs(shape=(3, 4), gapPenalty=-1, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=True)

    expected = np.array([
        ['', LEFT_DIR, LEFT_DIR, LEFT_DIR],
        [UP_DIR,   '',     '',      ''],
        [UP_DIR,   '',     '',      '']
    ], dtype='S9')

    assert np.array_equal(matrix, expected)

def test_createMatrix_DirectionsSq():
    args = MockArgs(shape=(3, 3), gapPenalty=-1, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=True)

    expected = np.array([
        ['', LEFT_DIR, LEFT_DIR],
        [UP_DIR,   '',     ''],
        [UP_DIR,   '',     '']
    ], dtype='S9')

    assert np.array_equal(matrix, expected)

def test_createMatrix_1x1Score():
    args = MockArgs((1, 1), gapPenalty=-3, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x1Direction():
    args = MockArgs((1, 1), gapPenalty=-3, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([['']], dtype="S9")
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Score():
    args = MockArgs((1, 4), gapPenalty=-2, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0], 
                        [-2],
                        [-4], 
                        [-6]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Score():
    args = MockArgs((1, 4), gapPenalty=-2, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=False)
    expected = np.array([[0, -2, -4, -6]], dtype=np.int32)
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Direction():
    args = MockArgs((1, 4), gapPenalty=-2, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([[''], 
                        [LEFT_DIR],
                        [LEFT_DIR], 
                        [LEFT_DIR]], dtype='S9')
    assert np.array_equal(matrix, expected)

def test_createMatrix_1x4Direction():
    args = MockArgs((1, 4), gapPenalty=-2, match=1, misMatch=-1)
    matrix = createMatrix(args, isDirectionMatrix=True)
    expected = np.array([['', LEFT_DIR, LEFT_DIR, LEFT_DIR]], dtype='S9')
    assert np.array_equal(matrix, expected)


# calculateAntiDiagonals
def test_calculateAntidiagonals_2x2():
    args = MockArgs(shape=(2, 2), gapPenalty=-1, match=1, misMatch=-1)
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(1, 0), (0, 1)],
        [(1, 1)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_2x5():
    args = MockArgs(shape=(2, 5), gapPenalty=-1, match=1, misMatch=-1)
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
    args = MockArgs(shape=(2, 2), gapPenalty=-1, match=1, misMatch=-1)
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],
        [(1, 0), (0, 1)],       
        [(1, 1)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_1x4():
    args = MockArgs(shape=(1, 4), gapPenalty=-1, match=1, misMatch=-1)
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(1, 0)],
        [(2, 0)],
        [(3, 0)]         
    ]
    assert result == expected

def test_calculateAntidiagonals_4x1():
    args = MockArgs(shape=(4, 1), gapPenalty=-1, match=1, misMatch=-1)
    result = calculateAntidiagonals(args)
    
    expected = [
        [(0, 0)],       
        [(0, 1)],
        [(0, 2)],
        [(0, 3)]         
    ]
    assert result == expected


# calculateSingleCellScore



# fillMatrix


# calculateAntiDiagonals


# traceback


# printDirectionMatrix