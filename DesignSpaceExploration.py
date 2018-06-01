#!/usr/bin/python
import numpy as np
import time

LayerSize  = np.array([376, 128, 64, 17])
PipeLimit  = np.array([ 48, 128, 64, 17])
#LayerSize  = np.array([111, 64, 32, 8])
#PipeLimit  = np.array([ 48, 64, 32, 8])
#LayerSize  = np.array([ 17, 32, 16, 6])
#PipeLimit  = np.array([ 17, 32, 16, 6])
#LayerSize  = np.array([ 11, 32, 16, 3])
#PipeLimit  = np.array([ 11, 32, 16, 3])


GoodSolutionThreshold = 10

DSPLimit   = 1570#1963
MinCycles  = 2**31-1
Solution   = []
DSPUsage   = 2**31-1
BRAMUsage  = 2**31-1

def NumCycles(p0, p1, p2, p3):
  NumPipes = np.array([p0, p1, p2, p3])
  BlkDim = -(-LayerSize // NumPipes)
  # Forward Propagation Cycles
  FwdSndCycleTime = max(BlkDim[0], BlkDim[2], 4)
  FwdCycles = BlkDim[0] + (BlkDim[1]-1)*FwdSndCycleTime + BlkDim[2]*BlkDim[3]
  # Backward Propagation Cycles
  BwdTrdCycleTime = max(BlkDim[2], 4)
  BwdFstCycleTime = max(BlkDim[0], BlkDim[2], 4)
  BwdCycles = (BlkDim[3]-1)*BwdTrdCycleTime + BlkDim[2] + BlkDim[1]*BwdFstCycleTime
  # Overall Cycles
  return max(FwdCycles, BwdCycles)

def NumDSP(p0, p1, p2, p3):
  quad = 4*p0*p1 + 8*p1*p2 + 7*p2*p3
  lin  = 19*p1 + 19*p2 + 2*p3
  return quad+lin

def NumBRAM(p0, p1, p2, p3):
  return 3*(p0*p1 + p1*p2 + p2*p3)

# Solve the Design Space Exploration Problem
tic = time.time()

# Find the Best Solution
for p0 in range(1, PipeLimit[0]+1):
  for p1 in range(1, PipeLimit[1]+1):
    for p2 in range(1, PipeLimit[2]+1):
      for p3 in range(1, PipeLimit[3]+1):
	# Check Constraint
#	if p0 % 4 != 0: break
	CurDSP = NumDSP(p0,p1,p2,p3)
	if CurDSP > DSPLimit: break
	# Compare against current best solution
        CurCycles = NumCycles(p0,p1,p2,p3)
        if CurCycles<MinCycles or ((CurCycles==MinCycles) and CurDSP<DSPUsage):
	  MinCycles = CurCycles
	  Solution  = np.array([p0, p1, p2, p3])
	  DSPUsage  = CurDSP
	  BRAMUsage = NumBRAM(p0, p1, p2, p3)

# Output Result
print('-------------------------------------------------------------------------------------------------')
print('LayerSize = [%2d, %2d, %2d, %2d]') % (LayerSize[0], LayerSize[1], LayerSize[2], LayerSize[3])
DSPUsage  = NumDSP(Solution[0], Solution[1], Solution[2], Solution[3])
BRAMUsage = NumBRAM(Solution[0], Solution[1], Solution[2], Solution[3])
BlockDim        = -(-LayerSize // Solution)
PaddedLayerSize = BlockDim * Solution
print('------- Best Solution ---------------------------------------------------------------------------')
print('Cycles = %d, Solution = [%2d, %2d, %2d, %2d], DSP = %4d, BRAM = %4d, PaddedLayerSize = [%2d, %2d, %2d, %2d], BlockDim = [%2d, %2d, %2d, %2d]') \
   % (MinCycles, Solution[0], Solution[1], Solution[2], Solution[3], DSPUsage, BRAMUsage, \
      PaddedLayerSize[0], PaddedLayerSize[1], PaddedLayerSize[2], PaddedLayerSize[3], \
      BlockDim[0], BlockDim[1], BlockDim[2], BlockDim[3])
print('------- Good Solutions --------------------------------------------------------------------------')

# Find Good Solutions
for p0 in range(1, PipeLimit[0]+1):
  for p1 in range(1, PipeLimit[1]+1):
    for p2 in range(1, PipeLimit[2]+1):
      for p3 in range(1, PipeLimit[3]+1):
	# Check Constraint
#	if p0 % 4 != 0: break
	CurDSP = NumDSP(p0,p1,p2,p3)
	if CurDSP > DSPLimit: break
        CurCycles = NumCycles(p0,p1,p2,p3)
        # Check is current solution is good
        if (CurCycles < MinCycles + GoodSolutionThreshold) and CurDSP < DSPUsage:
	  BlkDim  = -(-LayerSize // np.array([p0, p1, p2, p3]))
	  PadLySz = BlkDim * np.array([p0, p1, p2, p3])
	  print('Cycles = %d, Solution = [%2d, %2d, %2d, %2d], DSP = %4d, BRAM = %4d, PaddedLayerSize = [%2d, %2d, %2d, %2d], BlockDim = [%2d, %2d, %2d, %2d]') \
	       % (CurCycles, p0, p1, p2, p3, CurDSP, NumBRAM(p0,p1,p2,p3), \
	          PadLySz[0], PadLySz[1], PadLySz[2], PadLySz[3], \
	          BlkDim[0], BlkDim[1], BlkDim[2], BlkDim[3])
	  
toc = time.time()

print('-------------------------------------------------------------------------------------------------')
print 'Elapsed Time = ' + repr(toc-tic) + ' seconds'
print('-------------------------------------------------------------------------------------------------')