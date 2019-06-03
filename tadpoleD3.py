import argparse
import os
import sys
from socket import gethostname
import time
import datetime




# don't change to from voxCommon import * as this could end up importing matplotlib
from voxCommon import addParserArgs

parser = argparse.ArgumentParser(description='Launches clustering model on '
                                             'using cortical thickness maps derived from MRI')

parser.add_argument('--runIndex', dest = 'runIndex', type = int, default = 1, help = 'index of run instance/process')

parser.add_argument('--nrProc', dest = 'nrProc', type = int, default = 1, help = '# of processes')

parser.add_argument('--modelToRun', dest = 'modelToRun', type = int, help = 'index of model to run')

parser.add_argument('--models', dest = 'models', help = 'index of first experiment to run')

parser.add_argument('--nrOuterIt', dest = 'nrOuterIt', type = int,
                    help = '# of outer iterations to run, for estimating clustering probabilities')

parser.add_argument('--nrInnerIt', dest = 'nrInnerIt', type = int,
                    help = '# of inner iterations to run, for fitting the model parameters and subj. shifts')

parser.add_argument('--nrClust', dest = 'nrClust', type = int, help = '# of clusters to fit')

parser.add_argument('--cluster', action = "store_true", default = False,
                    help = 'need to include this flag if runnin on cluster')

parser.add_argument('--agg', dest = 'agg', type = int, default = 0,
                    help = 'agg=1 => plot figures without using Xwindows, for use on cluster where the plots cannot be displayed '
                           ' agg=0 => plot with Xwindows (for use on personal machine)')

parser.add_argument('--rangeFactor', dest = 'rangeFactor', type = float,
                    help = 'factor x such that min -= rangeDiff*x/10 and max += rangeDiff*x/10')

parser.add_argument('--informPrior', dest = 'informPrior', type = int, default = 0,
                    help = 'enables informative prior based on gamma and gaussian dist')

parser.add_argument('--reduceSpace', dest = 'reduceSpace', type = int, default = 1,
                    help = 'choose not to save certain files in order to reduce space')

parser.add_argument('--alphaMRF', dest = 'alphaMRF', type = int, default = -1,
                    help = 'not used in this model')

parser.add_argument('--initClustering', dest = "initClustering", default = 'k-means',
                    help = 'initial clustering method: k-means or hist')

parser.add_argument('--leaderboard', dest = "leaderboard", type=int,
                    help = 'set 1 for leaderboard prediction, otherwise 0')

<<<<<<< HEAD
=======
parser.add_argument('--stdGammaAlpha', dest='stdGammaAlpha', type=float, default=0.0025,
                    help='std deviation of gamma prior on alpha')

parser.add_argument('--stdBeta', dest='stdBeta', type=float, default=0.1,
                    help='std deviation of gaussian prior on beta')
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358

args = parser.parse_args()

# don't import matplotlib until here, add other imports below
if args.agg:
  # print(matplotlib.__version__)
  import matplotlib
  # print(matplotlib.get_backend())
  matplotlib.use('Agg')
  # print(matplotlib.get_backend())
  # print(asds)

from voxCommon import *
<<<<<<< HEAD
import evaluationFramework, drcDEM, adniDEM
=======
import evaluationFramework
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
from voxelDPM import *
from aux import *
from adniCommon import *
from env import *
import pandas as pd
import PlotterVDPM

params, plotTrajParams = initCommonVoxParams(args)


plotTrajParams['legendCols'] = 4
plotTrajParams['diagColors'] = {CTL:'b', MCI:'g', AD:'r', -1:'y'}
plotTrajParams['diagLabels'] = {CTL:'CTL', MCI:'MCI', AD:'AD', -1:'N/A'}
plotTrajParams['ylimitsRandPoints'] = (-5,5)
plotTrajParams['diagNrs'] = [CTL, MCI, AD]

plotTrajParams['SubfigClustMaxWinSize'] = (1300, plotTrajParams['SubfigClustMaxWinSize'][1])
plotTrajParams['Clust3DMaxWinSize'] = (900, 600)
# plotTrajParams['ylimTrajWeightedDataMean'] = (-3,2)
plotTrajParams['ylimTrajSamplesInOneNoData'] = (-2.5,1.5)
plotTrajParams['biomkAxisLabel'] = 'Cortical Thickness Z-score'
plotTrajParams['biomkWasInversed'] = False

refDate = datetime.date(2000, 1, 1)

def cleanTadpoleData(df):

  df.loc[df.RAVLT_learning < 0, 'RAVLT_learning'] = np.nan
  df.loc[df.RAVLT_forgetting < 0, 'RAVLT_forgetting'] = np.nan
  df.loc[df.RAVLT_perc_forgetting < 0, 'RAVLT_perc_forgetting'] = np.nan


  petCols = list(df.loc[:, 'HIPPL01_BAIPETNMRC_09_12_16' : 'MCSUVRCERE_BAIPETNMRC_09_12_16'])
  # df[petCols].replace({'-4': np.nan, -4: np.nan}, inplace=True)
  for c in petCols:
    df.loc[df[c] == '-4', c] = np.nan
  return df

def cleanTadpoleDataD3(df):


  # print('df', df)
  # print('np.isnan(df.ADAS13', np.sum(np.isnan(df.FLDSTRENG_UCSFFSX_11_02_15_UCSFFSX51_08_01_16)))
  #

  # it seems like there is nothing to do. all NaNs are noaded properly

  return df

def dateDiffToMonths(diff):
  return diff.days / (365.0 / 12)

def parseTadpoleData(df):

  cols = list(df.loc[:, 'FDG':'EcogSPTotal']) + list(df.loc[:, 'Ventricles':'MidTemp']) \
     + list(df.loc[:, 'ST101SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16':'ST9SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16']) \
     + list(df.loc[:, 'ST101SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16':'ST9SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']) \
     + list(df.loc[:, 'HIPPL01_BAIPETNMRC_09_12_16':'MCSUVRCERE_BAIPETNMRC_09_12_16']) \
     + list(df.loc[:, 'CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16':'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16']) \
     + list(df.loc[:, 'CEREBELLUMGREYMATTER_UCBERKELEYAV1451_10_17_16':'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV1451_10_17_16']) \
     + list(df.loc[:, 'FA_CST_L_DTIROI_04_30_14':'AD_SUMFX_DTIROI_04_30_14']) \
     + list(df.loc[:, 'ABETA_UPENNBIOMK9_04_19_17':'PTAU_UPENNBIOMK9_04_19_17'])


  # TODO: re-process data more, continue form here: change AV45 -> AV45/SIZE of ROI
  print('cols', cols)
  # filter out the FS cols with Standard deviation of volumes, cort thickness, etc ... Only keep average
  colsFilt = []
  for col in cols:
    if col[:2] == 'ST' and (col[5] == 'S' or col[6] == 'S'):
      continue

    colsFilt += [col]


  # print(ads)
  # print(df.D1)
  # print(df.shape)
  d2Ind = df.RID[df.loc[:,'D2'] == 1].as_matrix()


  # actually, don't remove any data because some D2 subjects only have 1 visit anyway
  # mask = df.loc[:,'D1'] == 1
  # df = df[mask]
  # df.reset_index(drop=True, inplace=True)
  # # print(df.shape)
  # df.reset_index()
  # df.reindex(index=range(df.shape[0]))

  df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
  pickle.dump(dict(df=df), open('tadpoleCleanDf_D12mD3.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  df = pickle.load(open('tadpoleCleanDf_D12mD3.npz', 'rb'))['df']

  # print(adsa)
  # normalise ventricles by ICV
  df['Ventricles'] = df['Ventricles'] / df['ICV']


  data = df.as_matrix(columns=cols)


  # convert diagnoses such as 'MCI to Dementia' to 'Dementia', etc ...
  # ctlDxchange = [1, 7, 9] mciDxchange = [2, 4, 8] adDxChange = [3, 5, 6]
  mapping = {1: CTL, 7: CTL, 9: CTL, 2: MCI, 4: MCI, 8: MCI, 3: AD, 5: AD, 6: AD}
  # df.replace({'DXCHANGE': mapping}, inplace=True)
  df['DXCHANGE'] = df['DXCHANGE'].map(mapping)
  diag = df['DXCHANGE'].as_matrix()

  examDates = df.EXAMDATE.as_matrix()
  df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], format="%Y-%m-%d")

  df.to_csv('tadpoleCleanDf_D12mD3.csv', index = False)
  # print(ads)
  dataDf = df[cols]


  # build numpy string array
  nrCols = len(cols)
  labels = np.ndarray((nrCols,), dtype='S100')
  for c in range(nrCols):
    labels[c] = cols[c]


  partCode = df.RID.as_matrix()

  unqPartCode = np.unique(partCode)
  nrUnqSubj = len(unqPartCode)

  ageAtScan = np.zeros(partCode.shape, np.float)
  scanTimepts = np.zeros(partCode.shape, np.float)

  for s in range(nrUnqSubj):
    subjRowsCurr = df.RID == unqPartCode[s]
    ageAtBlCurr = df.AGE[subjRowsCurr]
    examDatesCurr = df.EXAMDATE[subjRowsCurr]

    minInd = np.argmin(examDatesCurr)
    yearsDiffs = [(d - examDatesCurr[minInd]).days/365 for d in examDatesCurr]

    ageAtScan[subjRowsCurr] = ageAtBlCurr + yearsDiffs

    scanTimepts[subjRowsCurr] = np.argsort(np.argsort(yearsDiffs))

    sortedVisitsCurr = np.argsort(yearsDiffs)
    diagCurrSorted = diag[subjRowsCurr][sortedVisitsCurr]

    notNanDiags = [d for d in diagCurrSorted if not np.isnan(d)]

    diagCurrSortedFilled = np.copy(diagCurrSorted)

    if len(notNanDiags) == 0:
      # set the subject diag as -1 if there is absolutely no diagnosis
      diagCurrSortedFilled[0] = -1
    else:
      if np.isnan(diagCurrSortedFilled[0]):
        diagCurrSortedFilled[0] = notNanDiags[0]

      for v in range(1, len(sortedVisitsCurr)):
        if np.isnan(diagCurrSortedFilled[v]):
          diagCurrSortedFilled[v] = diagCurrSortedFilled[v-1]


    diagFilledInOrigOrder = diagCurrSortedFilled[np.argsort(sortedVisitsCurr)]

    diag[subjRowsCurr] = diagFilledInOrigOrder

  # compute number of months since Jan 2000 for each EXAMDATEs
  monthsSinceRefTime = np.zeros(partCode.shape, np.float)

  for r in range(df.RID.shape[0]):
    monthsSinceRefTime[r] = dateDiffToMonths(df.EXAMDATE[r].date() - refDate)


  assert not np.isnan(ageAtScan).any()
  assert not np.isnan(diag).any()
  assert not np.isnan(scanTimepts).any()
  assert not np.isnan(partCode).any()
  assert not np.isnan(monthsSinceRefTime).any()
  # assert not np.isnan(examDates).any()

  return data, diag, labels, scanTimepts, partCode, ageAtScan, dataDf, \
    monthsSinceRefTime, examDates, d2Ind

def makeBiomksDecr(data, diag, labels):

  assert(data.shape[0] == diag.shape[0])

  # perform t-test on every voxel, sort them by p-values
  pVals = scipy.stats.ttest_ind(data[diag == CTL,:], data[diag == AD,:], nan_policy='omit')[1]

  sortedInd = np.argsort(pVals)
  print('sortedInd', sortedInd)

  print('data[diag == CTL, :]', data[diag == CTL, :])
  meanCTL = np.nanmean(data[diag == CTL, :], axis=0)
  meanAD =  np.nanmean(data[diag == AD, :], axis=0)
  stdCTL = np.nanstd(data[diag == CTL, :], axis=0)
  stdAD = np.nanstd(data[diag == AD, :], axis=0)

  # record which biomarkers have had their sign flipped. Multiply this vector
  # with the scale from the normalisation with controls that we did earlier.
  biomkScaleExtra = np.ones(pVals.shape)

  for b in sortedInd:

    if (pVals[b] < 0.001) and meanAD[b] > meanCTL[b]:
      data[:,b] = data[:,b] * (-1)
      biomkScaleExtra[b] = -1
      print('flipped sign for %s' % labels[b])

  return data, sortedInd, biomkScaleExtra, pVals

def visTadpoleHist(data, diag, age, labels, plotTrajParams, sortedByPvalInd):
  '''
  Plots average biomarker value for various ROIs

  :param data: NR_CROSS_SUBJ x NR_BIOMK array
  :param diag: NR_CROSS_SUBJ x 1
  :param age:  NR_CROSS_SUBJ x 1
  :param plotTrajParams: dictionary of plotting parameters
  :param sortedByPvalInd: ROI indicesof each point on the surface, sorted by p-value (the regions for which we observe the highest differences between CTL and AD apprear first)

  :return: figure handle
  '''


  fig = pl.figure()
  nrRows = 3
  nrCols = 4
  nrBiomkToDisplay = nrRows * nrCols

  nrSubj, nrBiomk = data.shape

  xs = np.linspace(np.min(age), np.max(age), 100)
  diagNrs = plotTrajParams['diagNrs']

  import VDPMNaN
  VDPMNaN.makeLongArray(data, scanTimepts, partCode, np.unique(partCode))

  for row in range(nrRows):
    for col in range(nrCols):
      b = row * nrCols + col # clusterNr
      print('Plotting biomk:', b)

      if b < nrBiomk:
        ax = pl.subplot(nrRows, nrCols, 1+np.mod(b, nrBiomkToDisplay))
        ax.set_title('b%d %s' % (b, labels[b][:10]))

        nnMask = np.logical_not(np.isnan(data[:,b]))
        dataNotNanS = data[nnMask,b]
        diagNotNanS = diag[nnMask]
        ageNotNanS = age[nnMask]

        print('dataNotNanS', dataNotNanS)
        print('diagNotNanS', diagNotNanS)

        for d in range(len(diagNrs)):
          ax.hist(dataNotNanS[diagNotNanS == diagNrs[d]],bins=20, alpha=0.5,
            label=plotTrajParams['diagLabels'][diagNrs[d]], color=plotTrajParams['diagColors'][diagNrs[d]])

        if col == 0:
          ax.set_ylabel('Z-score')

        if row == (nrRows - 1):
          ax.set_xlabel('biomk value')
        else:
          ax.set_xticks([])

        if b == 0:
          adjustCurrFig(plotTrajParams)
          fig.suptitle('indiv points', fontsize=20)

          h, axisLabels = ax.get_legend_handles_labels()

          legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )

          mng = pl.get_current_fig_manager()
          mng.resize(*plotTrajParams['SubfigVisMaxWinSize'])

  pl.show()

  return fig


def visTadpoleSpagetti(data, diag, age, scanTimepts, partCode, labels, plotTrajParams, sortedByPvalInd):
  '''
  Plots average biomarker value for various ROIs

  :param data: NR_CROSS_SUBJ x NR_BIOMK array
  :param diag: NR_CROSS_SUBJ x 1
  :param age:  NR_CROSS_SUBJ x 1
  :param plotTrajParams: dictionary of plotting parameters
  :param sortedByPvalInd: ROI indicesof each point on the surface, sorted by p-value (the regions for which we observe the highest differences between CTL and AD apprear first)

  :return: figure handle
  '''


  fig = pl.figure()
  nrRows = 3
  nrCols = 4
  nrBiomkToDisplay = nrRows * nrCols

  nrSubj, nrBiomk = data.shape

  xs = np.linspace(np.min(age), np.max(age), 100)
  diagNrs = plotTrajParams['diagNrs']
  import VDPMNan
  unqPartCode = np.unique(partCode)
  longData = VDPMNan.VDPMNan.makeLongArray(None, data, scanTimepts, partCode, unqPartCode)
  longDiag = VDPMNan.VDPMNan.makeLongArray(None, diag, scanTimepts, partCode, unqPartCode)
  longAge = VDPMNan.VDPMNan.makeLongArray(None, age, scanTimepts, partCode, unqPartCode)
  nrLongSubj = len(longDiag)

  for row in range(nrRows):
    for col in range(nrCols):
      b = row * nrCols + col # clusterNr
      print('Plotting biomk:', b)

      if b < nrBiomk:
        ax = pl.subplot(nrRows, nrCols, 1+np.mod(b, nrBiomkToDisplay))
        ax.set_title('b%d %s' % (b, labels[b][:10]))

        nnMask = np.logical_not(np.isnan(data[:,b]))
        dataNotNanS = data[nnMask,b]
        diagNotNanS = diag[nnMask]
        ageNotNanS = age[nnMask]

        # print('dataNotNanS', dataNotNanS)
        # print('diagNotNanS', diagNotNanS)

        for s in range(nrLongSubj):
          # print('longAge[s]', longAge[s])
          # print('longData[s][:,b]', longData[s][:,b])
          # print('longDiag[s][0]', longDiag[s][0])
          pl.plot(longAge[s], longData[s][:,b], c=plotTrajParams['diagColors'][longDiag[s][0]],
            label=plotTrajParams['diagLabels'][longDiag[s][0]])

        if col == 0:
          ax.set_ylabel('biomarker')

        if row == (nrRows - 1):
          ax.set_xlabel('age')
        else:
          ax.set_xticks([])

        if b == 0:
          adjustCurrFig(plotTrajParams)
          fig.suptitle('indiv points', fontsize=20)

          h, axisLabels = ax.get_legend_handles_labels()

          legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )

          mng = pl.get_current_fig_manager()
          mng.resize(*plotTrajParams['SubfigVisMaxWinSize'])

  pl.show()

  return fig

def launchTadpole(runIndex, nrProcesses, modelToRun):

<<<<<<< HEAD
  d3File = '../data/ADNI/challenge_training_data/neil_repo/TADPOLE_D3.csv'
  d3Df = pd.read_csv(d3File, low_memory = False)
  d3Df = cleanTadpoleDataD3(d3Df)

  doProcess = 0

  if doProcess:
=======
  d3File = 'TADPOLE_D3.csv'
  d3Df = pd.read_csv(d3File, low_memory = False)
  d3Df = cleanTadpoleDataD3(d3Df)

  genProcessedDataset = 1

  if genProcessedDataset:
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
    if args.leaderboard == 0:

      sys.stdout.flush()
      outFileCheckpoint2 = 'tadpoleD3cp2.npz'
      print('loading data file')

<<<<<<< HEAD
      d1d2File = '../data/ADNI/challenge_training_data/neil_repo/TADPOLE_D1_D2.csv'
=======
      d1d2File = 'TADPOLE_D1_D2.csv'
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
      d12Df = pd.read_csv(d1d2File,low_memory=False)
      d12Df = cleanTadpoleData(d12Df)

      # filter out the RIDs from D1D2 which are in D3
      filtermask = np.logical_not(np.in1d(d12Df.RID, d3Df.RID))
      print('filtermask', np.sum(filtermask), filtermask.shape)
      print('old shape', d12Df.shape)
      d12Df = d12Df[filtermask]
      d12Df.reset_index(drop=True, inplace=True)
      d12Df.reindex(index=range(d12Df.shape[0]))

      # print('new shape', d12Df.shape)
      # print(adsa)
      # now train model on {D12 - D3}
      data, diag, labels, scanTimepts, partCode, ageAtScan, dataDf, monthsSinceRefTime, \
        examDates, predInd = parseTadpoleData(d12Df)

    else:
      #TODO
      raise ValueError('leaderboard not implemented for D3 yet')


    dataStruct = dict(data=data, diag=diag, labels=labels, scanTimepts=scanTimepts,
      partCode=partCode, ageAtScan=ageAtScan, dataDf=dataDf,
      monthsSinceRefTime=monthsSinceRefTime, examDates=examDates, predInd=predInd)
    pickle.dump(dataStruct, open(outFileCheckpoint2, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  else:
    if args.leaderboard == 0:
      outFileCheckpoint2 = 'tadpoleD3cp2.npz'
    else:
      outFileCheckpoint2 = 'tadpoleD3cp2Ldb.npz'


  dataStruct = pickle.load(open(outFileCheckpoint2, 'rb'))
  data = dataStruct['data']
  diag = dataStruct['diag']
  labels = dataStruct['labels']
  scanTimepts = dataStruct['scanTimepts']
  partCode = dataStruct['partCode']
  ageAtScan = dataStruct['ageAtScan']
  # dataDf = dataStruct['dataDf']
  monthsSinceRefTime = dataStruct['monthsSinceRefTime']
  examDates = dataStruct['examDates']
  predInd = dataStruct['predInd']


  # filter AD subjects
  # diagInd = np.array(np.where(matData['diag'] == PCA)[0])
  print('compiling parameters')
  sys.stdout.flush()

  print('diag', np.unique(diag), diag)
  # print(adsas)

  unqPartCode = np.unique(partCode)
  nrUnqPart = len(unqPartCode)

  # calculate Z-scores at each point w.r.t controls at baseline
  # controlBlInd = np.logical_and(diag == CTL, scanTimepts == 1)
  controlInd = diag == CTL
  stdBiomk = np.nanstd(data[diag == CTL], 0)
  biomkMaskCTL = np.isnan(np.nanstd(data[diag == CTL], 0))
  biomkMaskAD = np.isnan(np.nanstd(data[diag == AD], 0))
  biomkMaskMCI = np.isnan(np.nanstd(data[diag == MCI], 0))
  mask = np.logical_or(np.logical_or(biomkMaskCTL, biomkMaskMCI), biomkMaskAD)
  # print(ads)
  selectedBiomk = np.logical_not(np.logical_or(mask, stdBiomk == 0))

  print(data.shape)
  data = data[:, selectedBiomk]
  labels = labels[selectedBiomk]
  pointIndices = np.array(range(data.shape[1]))
  stdBiomk = np.nanstd(data[controlInd], 0)
  print(data.shape)
  # print(ads)

  meanCTL = np.nanmean(data[controlInd], 0)  # calculate Z-scores
  stdCTL = np.nanstd(data[controlInd], 0)
  dataZ = (data - meanCTL[None,:])/stdCTL[None,:]
  data = dataZ

  outlierRows, outlierCols = np.where(np.abs(dataZ) > 50)
  filterMask = np.ones(data.shape[0], bool)
  filterMask[outlierRows] = 0
  data = data[filterMask]
  diag = diag[filterMask]
  scanTimepts = scanTimepts[filterMask]
  partCode = partCode[filterMask]
  ageAtScan = ageAtScan[filterMask]
<<<<<<< HEAD
  monthsSinceRefTime = monthsSinceRefTime[filterMask]
  examDates = examDates[filterMask]
=======

  monthsSinceRefTime = monthsSinceRefTime[filterMask]
  examDates = examDates[filterMask]
  meanAgeAtScan = np.mean(ageAtScan.astype(float))
  ageAtScanCentered = (ageAtScan - meanAgeAtScan).astype(np.float16)
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358

  nrSubj, nrBiomk = data.shape
  # print('nrBiomk', nrBiomk)
  # print(adsa)

  dataAD = data[diag == AD, :]

  # make all biomarkers decreasing by flipping their signs if necessary
  # also perform a t-test to see which ones are most informative, sort them by pvalue (i.e. sortedByPvalInd)
  # the new data is re-scaled
  data, sortedByPvalInd, biomkScaleExtra, pVals = makeBiomksDecr(data, diag, labels)
  #doTtest(data, diag, pointIndices)

  # multiply the scaling we did from controls with (-1) if the biomk had the sign flipped
  stdBiomkRescale = biomkScaleExtra * stdCTL

  assert(sortedByPvalInd.shape[0] == data.shape[1])

  sys.stdout.flush()

  global params

  params['data'] = data
  params['diag'] = diag
  params['scanTimepts'] = scanTimepts
  params['partCode'] = partCode
  params['ageAtScan'] = ageAtScan
<<<<<<< HEAD
=======
  params['initShift'] = ageAtScanCentered # initialise time shifts (betas) to (age - meanAge)
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
  params['biomkDir'] = DECR
  params['modelToRun'] = modelToRun
  params['datasetFull'] = 'tadpole'
  params['labels'] = labels
  params['examDates'] = examDates

  # filter down to 100 subjects to make it run faster, just for testing. Also select only some biomarkers
  unqPartCode = np.unique(params['partCode'])
  nrPartToSample = 100
  np.random.seed(3)
  selectedPartCode = np.random.choice(unqPartCode, nrPartToSample)
  dataIndices = np.in1d(params['partCode'], selectedPartCode)
  # params = filterDDSPAIndices(params, dataIndices)


  indices = [i for i in range(len(labels)) if labels[i] in
      [b'FDG', b'AV45', b'MMSE', b'CDRSB', b'ADAS13', b'Ventricles',
       b'Hippocampus', b'WholeBrain', b'Entorhinal', b'MidTemp', b'ABETA_UPENNBIOMK9_04_19_17',
       b'TAU_UPENNBIOMK9_04_19_17', b'PTAU_UPENNBIOMK9_04_19_17']]

  # indices = sortedByPvalInd[:300]
  # print('pVals lowest', pVals[sortedByPvalInd[:300]])
  # print('pVals highest', pVals[sortedByPvalInd[-100:]])
  # print('indices', indices)
  # print(ads)
  print('labels', labels[indices])
  # print(adsa)
  print(np.nanstd(data,axis=0)[indices])
  data = params['data'][:,indices]
  params['data'] = data
  labels = labels[indices]
  params['labels'] = labels
  nrBiomk = params['data'].shape[1]
  print('data.shape', params['data'].shape)
  meanCTL = meanCTL[indices]
  stdBiomkRescale = stdBiomkRescale[indices]
  print(stdBiomkRescale)
  print('flippedBiomk', labels[stdBiomkRescale < 0])
  sortedByPvalInd = np.argsort(np.argsort(sortedByPvalInd[indices]))

  # visTadpoleHist(data, diag, ageAtScan, labels, plotTrajParams, sortedByPvalInd)
  # print(adsa)

  # visTadpoleSpagetti(data, diag, ageAtScan, scanTimepts, partCode, labels, plotTrajParams, sortedByPvalInd)
  # print(adsa)

  # print('CTL %f +/- %f', np.nanmean(params['data'][params['diag'] == CTL, 1]), np.nanstd(params['data'][params['diag'] == CTL, 1]))
  # print('AD %f +/- %f', np.nanmean(params['data'][params['diag'] == AD, 1]), np.nanstd(params['data'][params['diag'] == AD, 1]))
  # print(ads)

  # map points that have been removed to the closest included points (nearestNeighbours).
  # also find the adjacency list for the MRF and another subset of 10k points for
  # initial clustering
  runPartNN = 'L'
  plotTrajParams['nearestNeighbours'] = np.array(range(nrBiomk))
  params['adjList'] = np.nan
  params['nearNeighInitClust'] = np.array(range(nrBiomk))
  params['initClustSubsetInd'] = np.array(range(nrBiomk))
  params['meanBiomkRescale'] = meanCTL # for rescaling back if necessary
  params['stdBiomkRescale'] = stdBiomkRescale
<<<<<<< HEAD
  params['fixSpeed'] = True # if true then don't model progression speed, only time shift
=======
  params['fixSpeed'] = False # if true then don't model progression speed, only time shift
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358

  diagNrs = np.unique(diag)
  # print('diagNrs, diag', diagNrs, diag)
  # print(asdas)

  # print(len(params['acqDate']), data.shape[0])
  sys.stdout.flush()
  assert(params['data'].shape[0] == params['diag'].shape[0] ==
    params['scanTimepts'].shape[0] == params['partCode'].shape[0] ==
    params['ageAtScan'].shape[0])

  # sets an uninformative or informative prior
  priorNr = setPrior(params, args.informPrior, mean_gamma_alpha=1,
<<<<<<< HEAD
    std_gamma_alpha=0.3, mu_beta=0, std_beta=5)
=======
    std_gamma_alpha=0.1, mu_beta=0, std_beta=5)
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358

  suffix = ''
  if args.leaderboard:
    suffix = 'Ldb'
    # print(ads)

  expName = 'tadpoleD3Init%sCl%dPr%dRa%d%s' % (args.initClustering, params['nrClust'],
    priorNr, args.rangeFactor, suffix)
  plotTrajParams['sortedByPvalInd'] = sortedByPvalInd
  plotTrajParams['pointIndices'] = pointIndices
  plotTrajParams['expName'] = expName
  plotTrajParams['ageTransform'] = (0, 1) # no age normalisation was necessary
  plotTrajParams['datasetFull'] = params['datasetFull']
  plotTrajParams['labels'] = labels

  params['plotTrajParams'] = plotTrajParams

  # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
<<<<<<< HEAD
  params['runPartStd'] = ['L', 'L', 'I', 'I', 'I']
  params['runPartMain'] = ['R', 'I', 'I']  # [mainPart, plot, stage]
=======
  params['runPartStd'] = ['R', 'R', 'I', 'I', 'I']
  params['runPartMain'] = ['R', 'I', 'I', 'I']  # [mainPart, plot, stage, globalMinStats]
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
  params['runPartCogCorr'] = ['I']
  params['runPartCogCorrMain'] = ['L', 'L', 'I', 'I', 'L']
  params['runPartDirDiag'] = ['R', 'R', 'I']
  params['runPartStaging'] = ['L', 'L', 'I']
  params['runPartDiffDiag'] = ['R', 'R', 'I']
  params['runPartConvPred'] = ['I', 'I', 'I']
  params['runPartCVNonOverlap'] = ['R']
  params['runPartCVNonOverlapMain'] = ['L', 'L', 'I', 'I', 'L']
  params['masterProcess'] = runIndex == 0

  if params['masterProcess']:
    # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
    params['runPartStd'] = ['L', 'L', 'I', 'I', 'I']
    params['runPartMain'] = ['R', 'R', 'R']  # [mainPart, plot, stage]
    params['runPartCogCorr'] = ['I']
    params['runPartCogCorrMain'] = ['L', 'L', 'I', 'I', 'I']
    params['runPartDirDiag'] = ['R', 'R', 'I']
    params['runPartStaging'] = ['L', 'L', 'I']
    params['runPartDiffDiag'] = ['R', 'R', 'I']
    params['runPartConvPred'] = ['I', 'I', 'I']
    params['runPartCVNonOverlap'] = ['I']
    params['runPartCVNonOverlapMain'] = ['R', 'R', 'I', 'R', 'R']

  runAllExpFunc = runAllExpTADPOLE
  modelNames, res = evaluationFramework.runModels(params, expName, modelToRun, runAllExpFunc)

  # now generate forecast
  print('Generating forecast ... ')
  teamName = 'DIVE6'
  if args.leaderboard:
    outputFile = 'TADPOLE_Submission_Leaderboard_D3_%s.csv' % teamName
    predStartDate = datetime.date(2010, 5, 1)
    nrYearsToPred = 7
    nrMonthsToPred = 12*nrYearsToPred  # 5 years
  else:
    outputFile = 'TADPOLE_Submission_D3_%s.csv' % teamName
    predStartDate = datetime.date(2018, 1, 1)
    nrYearsToPred = 5
    nrMonthsToPred = 12*nrYearsToPred  # 7 years

  resCurrModel = res[0]['std']
  dpmObj = res[0]['dpmObj']


  nrSubjD3 = d3Df.shape[0]
  dataD3inD12format = np.nan * np.ones((nrSubjD3, data.shape[1]), float)
  print('labels', labels)
  print('cols', d3Df.columns)
  d3Df['Ventricles'] = d3Df.Ventricles / d3Df.ICV

# labels [b'FDG' b'AV45' b'CDRSB' b'ADAS13' b'MMSE' b'Ventricles' b'Hippocampus'
#  b'WholeBrain' b'Entorhinal' b'MidTemp' b'ABETA_UPENNBIOMK9_04_19_17'
#  b'TAU_UPENNBIOMK9_04_19_17' b'PTAU_UPENNBIOMK9_04_19_17']

  # dataD3inD12format[:, 2] = d3Df.CDRSOB
  dataD3inD12format[:, 3] = d3Df.ADAS13
  dataD3inD12format[:, 4] = d3Df.MMSE
  dataD3inD12format[:, 5] = d3Df.Ventricles
  dataD3inD12format[:, 6] = d3Df.Hippocampus
  dataD3inD12format[:, 7] = d3Df.WholeBrain
  dataD3inD12format[:, 8] = d3Df.Entorhinal
  dataD3inD12format[:, 9] = d3Df.MidTemp

  # there is no data for FDG, AV45 and CSF biomarkers, so leave them all as NaNs.

  # normalise D3 data to Z-scores
  dataD3inD12format = (dataD3inD12format - params['meanBiomkRescale'][None, :]) / params['stdBiomkRescale'][None, :]

  ageAtScanD3 = d3Df.AGE.as_matrix()
  partCodeD3 = d3Df.RID.as_matrix()
  print('partCodeD3', partCodeD3[:10])
  print('ageAtScanD3', ageAtScanD3[:10])
  for s in range(nrSubjD3):
    if d3Df.VISCODE[s][0] == 'm':
      ageAtScanD3[s] += float(d3Df.VISCODE[s][1:])/12

  params['predInd'] = partCodeD3

  d3Df['EXAMDATE'] = pd.to_datetime(d3Df['EXAMDATE'], format = "%Y-%m-%d")
  examDatesD3 = d3Df['EXAMDATE']

  mapping = {'NL': CTL, 'MCI': MCI, 'Dementia': AD, 'NL to MCI': MCI, 'MCI to Dementia':AD,
  'MCI to NL': CTL, 'Dementia to MCI':MCI}
  d3Df.replace({'DX': mapping}, inplace = True)
  diagD3 = d3Df['DX']

  # print(np.nanmean(dataD3inD12format[diagD3 == CTL,:],axis=0))
  # print(np.nanmean(data[diag == CTL], axis = 0))
  # print(adsa)

  # convert to Masked array
  dataD3inD12format = np.ma.masked_array(dataD3inD12format, np.isnan(dataD3inD12format))

    # print('ageAtScanD3', ageAtScanD3[:10])
  # print(adsa)
  # dataD3inD12format

  predAdasAllSubj, predVentsAllSubj, predDiagAllSubj = makeTadpoleForecastD3(predStartDate,
    nrYearsToPred, nrMonthsToPred, resCurrModel, params, dataD3inD12format, ageAtScanD3, partCodeD3,
    examDatesD3, dpmObj)

  # write forecast to file
  writeTadpoleSubmission(predAdasAllSubj, predVentsAllSubj, predDiagAllSubj, outputFile,
    nrMonthsToPred, predStartDate, params)


def makeTadpoleForecastD3(predStartDate, nrYearsToPred, nrMonthsToPred, resCurrModel, params,
  dataD3inD12format, ageAtScanD3, partCodeD3, examDatesD3, dpmObj):

  yearsFromPredStartToEachPredDate = np.linspace(0, nrYearsToPred, num=nrMonthsToPred, endpoint=False)

  nrClust = params['nrClust']
  assert abs(yearsFromPredStartToEachPredDate[1] - (1.0/12)) < 0.00001
  # make predictions
  startMonth = dateDiffToMonths(refDate - predStartDate)

  trajFunc = sigmoidFunc

  unqPartCodeFromRes = resCurrModel['uniquePartCode']

  nrSubjPredSet = partCodeD3.shape[0]
  # for each patient
  clustProbBC = resCurrModel['clustProb']
  thetas = resCurrModel['thetas']
  variances = resCurrModel['variances']

  labels = params['labels']
  indexAdas = np.where(labels == b'ADAS13')[0][0]
  indexVents = np.where(labels == b'Ventricles')[0][0]

  predDiagAllSubj = np.zeros((nrSubjPredSet, nrMonthsToPred, 3), np.float)
  predAdasAllSubj = np.zeros((nrSubjPredSet, nrMonthsToPred, 3), np.float)
  predVentsAllSubj = np.zeros((nrSubjPredSet, nrMonthsToPred, 3), np.float)

  dpsCross = resCurrModel['dpsCross']
  crossDiag = resCurrModel['crossDiag']

  dpsCTL = dpsCross[crossDiag == CTL]
  dpsMCI = dpsCross[crossDiag == MCI]
  dpsAD = dpsCross[crossDiag == AD]

  data = params['data']


  kernelWidth = np.std(dpsCross)/6 # need to test this parameter by visualisation

  from sklearn.neighbors.kde import KernelDensity
  kdeCTL = KernelDensity(kernel = 'gaussian', bandwidth = kernelWidth).fit(dpsCTL.reshape(-1,1))
  kdeMCI = KernelDensity(kernel = 'gaussian', bandwidth = kernelWidth).fit(dpsMCI.reshape(-1,1))
  kdeAD = KernelDensity(kernel = 'gaussian', bandwidth = kernelWidth).fit(dpsAD.reshape(-1,1))

  kdeXs = np.linspace(np.min(dpsCross), np.max(dpsCross), num=100).reshape(-1,1)

  fig = pl.figure(3)
  pl.clf()
  print('kdeCTL.score_samples(kdeXs)', np.exp(kdeCTL.score_samples(kdeXs)))
  pl.plot(kdeXs, np.exp(kdeCTL.score_samples(kdeXs)), label='CTL', c=plotTrajParams['diagColors'][CTL])
  pl.plot(kdeXs, np.exp(kdeMCI.score_samples(kdeXs)), label='MCI', c=plotTrajParams['diagColors'][MCI])
  pl.plot(kdeXs, np.exp(kdeAD.score_samples(kdeXs)),  label='AD', c=plotTrajParams['diagColors'][AD])
  pl.legend()
  pl.xlabel('disease progression score')
  pl.ylabel('likelihood')
  fig.show()
  fig.savefig('%s/diagHistD3.png' % (resCurrModel['outFolder']), dpi=100)

  runPred = 'R'
  doPlot = 0
  predFile = 'tadpolePredD3.npz'

  meanBiomkRescale = params['meanBiomkRescale']
  stdBiomkRescale = params['stdBiomkRescale']

  subShiftsD3, dpsCrossD3 = dpmObj.stageSubjectsCrossDataAge(dataD3inD12format, ageAtScanD3)
  ds=dict(subShiftsD3=subShiftsD3, dpsCrossD3=dpsCrossD3)
  pickle.dump(ds, open('d3Stages.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
<<<<<<< HEAD
  print(adsa)
=======
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358

  ds = pickle.load(open('d3Stages.npz', 'rb'))
  subShiftsD3 = ds['subShiftsD3']
  dpsCrossD3 = ds['dpsCrossD3']

  print('dpmObj.subShifts', dpmObj.subShifts)
  print('subShiftsD3', subShiftsD3)
  print('dpsCrossD3', dpsCrossD3)
  # print(adsas)


  if runPred == 'R':
    for s in range(nrSubjPredSet):

      ######### find dps at forecasted months ##########

      # find age at forecasted months
      subjRowsCurr = partCodeD3 == partCodeD3[s]

      # import pdb
      # pdb.set_trace()

      # for one timepoint, find the age and the examDate
      print('part : ', partCodeD3[s], np.sum(subjRowsCurr))
      print('part ageAtScanD3: ', partCodeD3[s], ageAtScanD3[subjRowsCurr][0])

      # compute age of subject at every prediction date
      ageOneTimept = ageAtScanD3[subjRowsCurr][0]
      examDateOneTimept = examDatesD3[subjRowsCurr].iloc[0].date()
      yearsToPredStartDate = (predStartDate - examDateOneTimept).days/365
      ageAtPredDates = ageOneTimept + yearsToPredStartDate + yearsFromPredStartToEachPredDate

      # compute dps
      dpsAtFutForecastDatesCurr = calcDpsGivenAges(ageAtPredDates, subShiftsD3[s,:])

      ######## find model predictions for those DPSs ##############3

      futureForecastsAdas, futureForecastsVents = calcModelPredAdasVents(dpsAtFutForecastDatesCurr,
      thetas, variances, clustProbBC[indexAdas, :].T, clustProbBC[indexVents, :].T, trajFunc)

      # add subject-specific intercept to the predictions, is subject has data
      # warning: can contain NaNs and even be NaN in all entries.
      adasDataCurrSubj = dataD3inD12format[subjRowsCurr, indexAdas]
      ventsDataCurrSubj = dataD3inD12format[subjRowsCurr, indexVents]

      ageCurrVisits = ageAtScanD3[subjRowsCurr]
      dpsSubjCurrVisits = dpsCrossD3[subjRowsCurr]
      currVisitsPredAdas, currVisitsPredVents = calcModelPredAdasVents(dpsSubjCurrVisits, thetas,
        variances, clustProbBC[indexAdas, :].T, clustProbBC[indexVents, :].T, trajFunc)

      futureForecastsAdas = addSubjIntercept(dpsAtFutForecastDatesCurr, futureForecastsAdas,
        adasDataCurrSubj, currVisitsPredAdas)
      futureForecastsVents = addSubjIntercept(dpsAtFutForecastDatesCurr, futureForecastsVents,
        ventsDataCurrSubj, currVisitsPredVents)

      # convert predictions back to un-normalised values

      predAdasNotNorm = futureForecastsAdas * stdBiomkRescale[indexAdas] + meanBiomkRescale[indexAdas]
      predVentsNotNorm = futureForecastsVents * stdBiomkRescale[indexVents] + meanBiomkRescale[indexVents]

      predAdasAllSubj[s, :, :] = predAdasNotNorm
      predAdasAllSubj[s, :, 1] = predAdasNotNorm[:,2] # need invert lower& upper bounds due to sign change
      predAdasAllSubj[s, :, 2] = predAdasNotNorm[:,1]

      predVentsAllSubj[s, :, :] = predVentsNotNorm
      predVentsAllSubj[s, :, 1] = predVentsNotNorm[:,2]
      predVentsAllSubj[s, :, 2] = predVentsNotNorm[:,1]

      ctlLik = np.exp(kdeCTL.score_samples(dpsAtFutForecastDatesCurr.reshape(-1,1)))
      mciLik = np.exp(kdeMCI.score_samples(dpsAtFutForecastDatesCurr.reshape(-1,1)))
      adLik = np.exp(kdeAD.score_samples(dpsAtFutForecastDatesCurr.reshape(-1,1)))

      sumLik = ctlLik + mciLik + adLik

      predDiagAllSubj[s, :, 0] = ctlLik/sumLik
      predDiagAllSubj[s, :, 1] = mciLik/sumLik
      predDiagAllSubj[s, :, 2] = adLik/sumLik

      adasDataCurrSubjUnnorm = adasDataCurrSubj * stdBiomkRescale[indexAdas] + meanBiomkRescale[indexAdas]
      ventsDataCurrSubjUnnorm = ventsDataCurrSubj* stdBiomkRescale[indexVents] + meanBiomkRescale[indexVents]

      if doPlot:
        if args.leaderboard:
          lb4Data = pd.read_csv('../data/ADNI/challenge_training_data/neil_repo/evaluation/TADPOLE_LB4.csv')
          lb4Data['CognitiveAssessmentDate'] = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in lb4Data['CognitiveAssessmentDate']]
          lb4Data['ScanDate'] = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in lb4Data['ScanDate']]
          mapping = {'CN': 0, 'MCI': 1, 'AD': 2}
          lb4Data.replace({'Diagnosis': mapping}, inplace=True)

          currSubjMaskLB4 = lb4Data.RID == partCodeD3[s]
          adasLB4CurrSubj = lb4Data.ADAS13[currSubjMaskLB4]
          ventsLB4CurrSubj = lb4Data.Ventricles[currSubjMaskLB4]
          diagLB4CurrSubj = lb4Data.Diagnosis[currSubjMaskLB4]

          datesLB4CurrSubj = lb4Data['CognitiveAssessmentDate'][currSubjMaskLB4]

          yearsFromRefDateToLB4Dates = np.array([(d.date() - examDateOneTimept).days/365 for d in datesLB4CurrSubj])
          ageAtLB4datesCurrSubj = ageOneTimept + yearsFromRefDateToLB4Dates

          lb4Params = dict(adasLB4CurrSubj=adasLB4CurrSubj, ventsLB4CurrSubj=ventsLB4CurrSubj,
            diagLB4CurrSubj=diagLB4CurrSubj, ageAtLB4datesCurrSubj=ageAtLB4datesCurrSubj)

        else:
          lb4Params = None

        plotSubjForecasts(predAdasAllSubj[s, :, :], predVentsAllSubj[s, :, :], predDiagAllSubj[s, :, :]
        , ageAtPredDates, adasDataCurrSubjUnnorm, ventsDataCurrSubjUnnorm, ageCurrVisits, lb4Params,
          rid=partCodeD3[s])

    ds = dict(predAdasAllSubj=predAdasAllSubj, predVentsAllSubj=predVentsAllSubj,
      predDiagAllSubj=predDiagAllSubj)
    pickle.dump(ds, open(predFile, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

  else:
    ds = pickle.load(open(predFile, 'rb'))
    predAdasAllSubj = ds['predAdasAllSubj']
    predVentsAllSubj = ds['predVentsAllSubj']
    predDiagAllSubj = ds['predDiagAllSubj']

  return predAdasAllSubj, predVentsAllSubj, predDiagAllSubj

def plotSubjForecasts(predAdasCurrSubj, predVentsCurrSubj, predDiagCurrSubj
  , ageAtPredDates, adasDataCurrSubjUnnorm, ventsDataCurrSubjUnnorm, ageCurrVisits, lb4Params, rid):

  if lb4Params is not None:
    adasLB4CurrSubj = lb4Params['adasLB4CurrSubj']
    ventsLB4CurrSubj = lb4Params['ventsLB4CurrSubj']
    diagLB4CurrSubj = lb4Params['diagLB4CurrSubj']
    ageAtLB4datesCurrSubj = lb4Params['ageAtLB4datesCurrSubj']

  pl.figure(3)
  ax = pl.subplot(1, 2, 1)
  ax.set_title('ADAS RID:%d' % rid)
  pl.plot(ageAtPredDates, predAdasCurrSubj)
  pl.scatter(ageCurrVisits, adasDataCurrSubjUnnorm, c='b',s=10)
  if lb4Params is not None:
    pl.scatter(ageAtLB4datesCurrSubj, adasLB4CurrSubj, c='r', s=10)

  ax = pl.subplot(1, 2, 2)
  ax.set_title('Vents RID:%d' % rid)
  pl.plot(ageAtPredDates, predVentsCurrSubj)
  pl.scatter(ageCurrVisits, ventsDataCurrSubjUnnorm, c='b', s=10)
  if lb4Params is not None:
    pl.scatter(ageAtLB4datesCurrSubj, ventsLB4CurrSubj, c='r', s=10)

  pl.show()


def calcModelPredAdasVents(dpsPredCurr, thetas, variances, clustProbAdas, clustProbVents, trajFunc):

  nrClust = thetas.shape[0]
  predCurrSubClustSC = np.zeros((dpsPredCurr.shape[0], nrClust), float)
  predCurrSubClustSClower = np.zeros((dpsPredCurr.shape[0], nrClust), float)
  predCurrSubClustSCupper = np.zeros((dpsPredCurr.shape[0], nrClust), float)

  for c in range(nrClust):
    predCurrSubClustSC[:, c] = trajFunc(dpsPredCurr, thetas[c, :])
    predCurrSubClustSClower[:, c] = predCurrSubClustSC[:, c] - 0.33 * np.sqrt(variances[c])
    predCurrSubClustSCupper[:, c] = predCurrSubClustSC[:, c] + 0.33 * np.sqrt(variances[c])

  # from the predictions of each cluster trajectories, predict traj of ADAS and Vents
  # using the probabilities of ADAS/Vents of being assigned to each cluster
  futureForecastsAdas = np.zeros((predCurrSubClustSC.shape[0],3))
  futureForecastsVents = np.zeros((predCurrSubClustSC.shape[0],3))

  futureForecastsAdas[:,0] = np.dot(predCurrSubClustSC, clustProbAdas)
  futureForecastsVents[:,0] = np.dot(predCurrSubClustSC, clustProbVents)

  futureForecastsAdas[:,1] = np.dot(predCurrSubClustSClower, clustProbAdas)
  futureForecastsVents[:,1] = np.dot(predCurrSubClustSClower, clustProbVents)

  futureForecastsAdas[:,2] = np.dot(predCurrSubClustSCupper, clustProbAdas)
  futureForecastsVents[:,2] = np.dot(predCurrSubClustSCupper, clustProbVents)

  return futureForecastsAdas, futureForecastsVents

def calcDpsGivenAges(ageAtPredDates, subShiftsCurr):

  subShiftsPredDates = np.tile(subShiftsCurr, (ageAtPredDates.shape[0], 1))

  print('subShiftsPredDates', subShiftsPredDates.shape)
  print('ageAtPredDates', ageAtPredDates.shape)
  assert subShiftsPredDates.shape[0] == ageAtPredDates.shape[0]
  assert subShiftsPredDates.shape[1] == 2

  dpsPredCurr = VoxelDPM.calcDpsNo1array(subShiftsPredDates, ageAtPredDates)

  return dpsPredCurr

def addSubjIntercept(dpsT, futurePredictions, dataCurrSubjT, modelPredExistingVisits):

<<<<<<< HEAD
  if np.isnan(dataCurrSubjT).all():
=======
  if np.isnan(dataCurrSubjT).all() or (dataCurrSubjT.count() == 0):
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
    # no data available cur current subject, leave as population estimate
    return futurePredictions
  else:
    # data is
<<<<<<< HEAD
=======
    # print('futurePredictions', futurePredictions)
    # print('dataCurrSubjT', type(dataCurrSubjT), dataCurrSubjT.count(), dataCurrSubjT)
    # print(np.nanmean(dataCurrSubjT))
    # print(np.mean(modelPredExistingVisits))
>>>>>>> a68c3519cc7626e8b66ebace63280668ff32a358
    return futurePredictions + (np.nanmean(dataCurrSubjT) - np.mean(modelPredExistingVisits))

def writeTadpoleSubmission(predAdasAllSubj, predVentsAllSubj, predDiagAllSubj, outputFile,
  nrMonthsToPred, predStartDate, params):

  predInd = params['predInd']
  predSetRidUnq = np.unique(predInd)
  print('Writing forecast to file %s' % outputFile)
  submission_table = pd.DataFrame()
  nrSubjPredSet = predSetRidUnq.shape[0]
  # * Repeated matrices - compare with submission template
  submission_table['RID'] = predSetRidUnq.repeat(nrMonthsToPred)
  submission_table['Forecast Month'] = np.tile(range(1, nrMonthsToPred + 1),
    (nrSubjPredSet, 1)).flatten()

  from dateutil.relativedelta import relativedelta
  endDate = predStartDate + relativedelta(months = +nrMonthsToPred - 1)
  ForecastDates = [predStartDate]
  while ForecastDates[-1] < endDate:
    ForecastDates.append(ForecastDates[-1] + relativedelta(months = +1))

  ForecastDatesStrings = [datetime.datetime.strftime(d, '%Y-%m') for d in ForecastDates]
  submission_table['Forecast Date'] = np.tile(ForecastDatesStrings, (nrSubjPredSet, 1)).flatten()
  # * Pre-fill forecast data, encoding missing data as NaN
  nanColumn = np.repeat(np.nan, submission_table.shape[0])
  submission_table['CN relative probability'] = nanColumn
  submission_table['MCI relative probability'] = nanColumn
  submission_table['AD relative probability'] = nanColumn
  submission_table['ADAS13'] = nanColumn
  submission_table['ADAS13 50% CI lower'] = nanColumn
  submission_table['ADAS13 50% CI upper'] = nanColumn
  submission_table['Ventricles_ICV'] = nanColumn
  submission_table['Ventricles_ICV 50% CI lower'] = nanColumn
  submission_table['Ventricles_ICV 50% CI upper'] = nanColumn

  # *** Paste in month-by-month forecasts **
  # * 1. Clinical status
  submission_table['CN relative probability'] = predDiagAllSubj[:, :, 0].flatten()
  submission_table['MCI relative probability'] = predDiagAllSubj[:, :, 1].flatten()
  submission_table['AD relative probability'] = predDiagAllSubj[:, :, 2].flatten()
  # * 2. ADAS13 score
  submission_table['ADAS13'] = predAdasAllSubj[:, :, 0].flatten()
  submission_table['ADAS13 50% CI lower'] = predAdasAllSubj[:, :, 1].flatten()
  submission_table['ADAS13 50% CI upper'] = predAdasAllSubj[:, :, 2].flatten()
  # * 3. Ventricles volume (normalised by intracranial volume)
  submission_table['Ventricles_ICV'] = predVentsAllSubj[:, :, 0].flatten()
  submission_table['Ventricles_ICV 50% CI lower'] = predVentsAllSubj[:, :, 1].flatten()
  submission_table['Ventricles_ICV 50% CI upper'] = predVentsAllSubj[:, :, 2].flatten()

  submission_table.to_csv(outputFile, index = False)



def runAllExpTADPOLE(params, expName, dpmBuilder):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = []
  params['excludeStaging'] = [-1]
  params['anchorID'] = MCI

  # run if this is the master   process or nrProcesses is 1
  unluckyProc = (np.mod(params['currModel'] - 1, params['nrProcesses']) == params['runIndex'] - 1)
  unluckyOrNoParallel = unluckyProc or (params['nrProcesses'] == 1) or params['masterProcess']

  if unluckyOrNoParallel:
    dpmObj, res['std'] = evaluationFramework.runStdDPM(params, expName, dpmBuilder, params['runPartMain'])

  res['dpmObj'] = dpmObj

  return res


def printResADNIthick(modelNames, res, plotTrajParams):
  #nrModels = len(modelNames)

  # dinamicModelName = 'VWDPMLinear'
  # staticModelName = 'VWDPMLinearStatic'
  # dinamicModelName = 'VDPM_MRF'
  # staticModelName = 'VWDPMStatic'
  # noDPSModelName = 'VDPMNoDPS'

  print('##### biomk prediction ######')
  nrModels = len(modelNames)
  pred = list(range(nrModels))
  predMean = list(range(nrModels))
  predStd = list(range(nrModels))
  for m in range(nrModels):
    pred[m] = res[m]['cogCorr']['predStats']
    predMean[m] = np.nanmean(pred[m])
    predStd[m] = np.nanstd(pred[m])

  for m in range(nrModels):
    print('%s predAllFolds' % modelNames[m], pred[m])
  for m in range(nrModels):
    print('%s predMean' % modelNames[m], predMean[m])
  for m in range(nrModels):
    print('%s predStd' % modelNames[m], predStd[m])

  stats = list(range(nrModels))
  print('##### correlation with cog tests ######')
  for m in range(nrModels):
    stats[m] = res[m]['cogCorr']['statsAllFolds']  # shape (NR_FOLDS, 2*NR_COG_TESTS)
    #print('stats:', stats[m])
    print(modelNames[m],end='')
    meanStats = np.nanmean(stats[m], 0)
    stdStats = np.nanstd(stats[m], 0)
    for i in range(meanStats.shape[0]):
      print('%.2f +/- %.2f' % (meanStats[i], stdStats[i]), end='')
    print('')

  plotScoresHist(scores = pred, labels=modelNames)

  nrCogStats = stats[0].shape[1]

  # perform paired t-test, as the same cross-validation folds have been used in both cases
  tStats = np.zeros(nrCogStats,float)
  pVals = np.zeros(nrCogStats,float)
  for t in range(nrCogStats):
    tStats[t], pVals[t] = scipy.stats.ttest_rel(stats[0][:,t], stats[0][:,t])

  # expInds = [dinIndex, staIndex, noDPSIndex]
  # printDiffs(expInds, res, modelNames)

  # testPredFPBDin = res[dinIndex]['cogCorr']['testPredPredFPB']
  # testPredFPBSta = res[staIndex]['cogCorr']['testPredPredFPB']
  # testPredFPBNoDPS = res[noDPSIndex]['cogCorr']['testPredPredFPB']
  # testPredDataFPB = res[noDPSIndex]['cogCorr']['testPredDataFPB']
  #
  # outDir = 'resfiles/%s' % plotTrajParams['expName']
  # os.system('mkdir -p %s' % outDir)
  #
  # for i in [0]: #range(len(testPredFPBDin)):
  #   print('testPred diff [%d] ' % i, testPredFPBDin[i] - testPredFPBSta[i])
  #
  #   meanAbsDiffB = np.sum(np.abs(testPredFPBDin[i] - testPredFPBSta[i]), axis=0)
  #   plotter = PlotterVDPM.PlotterVDPM()
  #   plotter.plotDiffs(meanAbsDiffB, plotTrajParams,
  #     filePathNoExt='%s/diffPredDinSta_f%d' % (outDir, i))



# def printDiffs(expInds, res, modelNames):
#
#   for ind in range(len(expInds)):




if __name__ == '__main__':
  # model 4 - VDPM sigmoidal
  # model 5 - VDPM linear

  if args.modelToRun:
    modelToRun = args.modelToRun
  elif args.models:
    modelToRun = np.array([int(i) for i in args.models.split(',')])
  else:
    raise ValueError('need to set either --models or --firstModel & --lastModel')

  launchTadpole(args.runIndex, args.nrProc, modelToRun)
