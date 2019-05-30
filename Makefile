

##### TADPOLE ########


# run on main prediction set D2
tadpoleD2:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 0 --agg 1

# run on cross-sectional prediction set D3
tadpoleD3:
	python3 tadpoleD3.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 13 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 0 --agg 1

# run on leaderboard
tadpoleLdb:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 1 --agg 1
