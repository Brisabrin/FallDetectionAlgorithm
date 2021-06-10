# FallDetectionAlgorithm

Currently developed AI model to differentiate daily activities with falling incidents for improved fall detection to be applied to the engineered device, wearable airbag for minimising fall impacts. 
Implemented Multivariate Classification on time series data - extracted from the SisFall Dataset ( a relatively large dataset ) . Furthermore, utilized the Sktime Library to test specific algorithms targetted/suitable  for Time Series data ( TimeSeriesForest , Random Interval Spectral Ensemble ( RISE ) , KNN with Dynamic time Warping ( DTW ) etc. ) 


Current results : 
- TimeSeriesForest Classifier managed to yield an accuracy of 97 % using no_estimators = 10 and 100 % using no_estimators = 100  ( although K cross fold Validation haven't been applied and have not tested it on other Fall Datasets ) 



