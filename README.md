## STE Model for Air Quality Prediction


Implement a novel and effective spatial-temporal ensemble model for air quality prediction. 

1. dat：Data of the model. 
	- air_bj.mat: Air quality and meteorology data of one year, sampled once an hour, from 35 monitoring stations in Beijing. Location information(longitude and latitude) of stations.
	
2. baselines: Baselines of our model.
	- Linear regression
	- Gaussian process
	- Regression tree
	- SVR, Neural network
	- Deep neural network
	- FFA model: The model proposed in [2].
	
3. lib: Some relevant functions of our STE model. 
	- granger_cause: Granger causality. To discover the spatial correlation. 
	- libsvm-master: SVM & SVR. Need to install.
	- myKmeans: K-Means and fc-Means. 
	- myNN: Neural network & Stack auto encoder.
	- myLSTM: LSTM.

4. src: Proposed STE model.
	- T.m: Only temporal part.
	- ST.m: Only spatial and temporal parts.
	- SE.m: Only spatial and ensemble parts.
	- STE.m: Spatial-temporal ensemble model.
	
Reliance:
1. [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 
	
Reference:

1. [Prediction as a candidate for learning deep hierarchical models of data](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6284) (Palm, 2012)
2. Y. Zheng, X. Yi, M. Li, R. Li, Z. Shan, E. Chang, T. Li, Forecasting fine-grained air quality based on big data, in: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM, 2015, pp. 2267–2276.
