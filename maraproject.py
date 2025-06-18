# dependencies
# numpy

import argparse
from ctypes import c_char, c_int
import numpy as np
from multiprocessing import Process, Array

LEFT_DIR = b'L'
UP_DIR   = b'U'
DIAG_DIR = b'D'

def createMatrix(args, *, isDirectionMatrix: bool):
    matrix = np.zeros(args.shape, dtype = "S3" if isDirectionMatrix else np.int32) 
    # seq1 = x = columns, seq2 = y = rows
    
    for x in range(args.shape[1]):
        matrix[0][x] = LEFT_DIR if isDirectionMatrix else x * args.gapPenalty
        # first row of the matrix, for each column add the gap penalty from the cell before

    for y in range(args.shape[0]):
        matrix[y][0] = UP_DIR if isDirectionMatrix else y * args.gapPenalty
        # for each row, in position 0 (first column), add the gap penalty from the cell above

    return matrix   
 

def fillMatrix(antidiag, args, scoreMatrix: np.ndarray, directionMatrix: np.ndarray):
    directionMatrix = Array(c_char, directionMatrix.tobytes())
    scoreMatrix = Array(c_int, scoreMatrix.flatten())

    for diag in antidiag:
        processes = []
        for cell in diag:
            p = Process(target=calculateSingleCellScore, args=(cell, args, scoreMatrix, directionMatrix))
            p.daemon = True
            p.start()
            processes.append(p)
       
        list(map(lambda p: p.join(), processes))

    return np.frombuffer(scoreMatrix.get_obj(), dtype=np.int32).reshape(args.shape), np.frombuffer(directionMatrix.get_obj(), dtype='S3').reshape(args.shape)

def calculateSingleCellScore(cell: tuple[int,int], args, scoreMatrix, directionMatrix) -> None:
    x, y = cell
    if y == 0 or x == 0:
        return

    scoreMatrix = np.frombuffer(scoreMatrix.get_obj(), dtype=np.int32).reshape(args.shape)
    directionMatrix = np.frombuffer(directionMatrix.get_obj(), dtype='S3').reshape(args.shape)

    upScore = scoreMatrix[y-1][x] + args.gapPenalty
    leftScore = scoreMatrix[y][x-1] + args.gapPenalty
    
    if args.seq1[x-1] == args.seq2[y-1]:
        diagScore = scoreMatrix[y-1][x-1] + args.match
    else:
        diagScore = scoreMatrix[y-1][x-1] + args.misMatch
    
    scoreMatrix[y][x] = max(upScore, leftScore, diagScore)

    cellDirections = b""

    if scoreMatrix[y][x] == upScore:
        cellDirections += UP_DIR
    if scoreMatrix[y][x] == leftScore:
        cellDirections += LEFT_DIR
    if scoreMatrix[y][x] == diagScore:
        cellDirections += DIAG_DIR

    directionMatrix[y][x] = cellDirections



def calculateAntidiagonals(args):
    rows, columns = args.shape 
    antiDiagList = [] # # each singleDiagList is inserted in this list with the last 'append'
    for index in range(columns):
        singleDiagList = [] # each diagonal is inserted in this list with the first 'append'
        x = index
        y = 0
        for _ in range(min(rows, index + 1)): 
            singleDiagList.append((x, y)) # here because i want to append the first cell too
            x -= 1
            y += 1
        
        antiDiagList.append(singleDiagList)

    for j in range(1, rows):
        singleDiagList2 = []
        x = columns - 1
        y = j
        for _ in range(min(columns, rows - j)):
            singleDiagList2.append((x, y))
            x -= 1
            y += 1

        antiDiagList.append(singleDiagList2)
    
    return antiDiagList


def traceback(directionMatrix, args):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Scientific Programming Project", description = "Takes two sequences as input for the allignement")
    parser.add_argument("--seq1", type=str, help="Write here your first sequence")
    parser.add_argument("--seq2", type=str, help="Write here your second sequence")
    parser.add_argument("-gp", "--gapPenalty", type=int, help="Write the negative gap penalty you want to apply")
    parser.add_argument("-m", "--match", type=int, help="Write the match score you want to apply")
    parser.add_argument("-mm", "--misMatch", type=int, help="Write the mismatch score you want to apply")

    args = parser.parse_args()
    args.shape = (len(args.seq2) + 1, len(args.seq1) + 1)

    scoreMatrix = createMatrix(args, isDirectionMatrix = False)
    directionMatrix = createMatrix(args, isDirectionMatrix = True)
    antidiag = calculateAntidiagonals(args)
    scoreMatrix, directionMatrix = fillMatrix(antidiag, args, scoreMatrix, directionMatrix)
    print(antidiag)
    print(scoreMatrix)
    print(directionMatrix)
    traceback = traceback(directionMatrix, args)
    print(traceback)


    print("First sequence:", args.seq1)
    print("Second sequence:", args.seq2)