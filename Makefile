

##### TADPOLE ########

tadpoleLdb:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 1 --agg 1

tadpoleD2:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 0 --agg 1

tadpoleD3:
	python3 tadpoleD3.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 13 --initClustering k-means --rangeFactor 1 --informPrior 0 --leaderboard 0 --agg 1

