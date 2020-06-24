package prea.recommender.llorma;

import prea.util.EvaluationMetrics;
import prea.util.KernelSmoothing;
import prea.recommender.Recommender;
import prea.recommender.matrix.RegularizedSVD;
import prea.data.structure.SparseVector;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import prea.data.structure.SparseMatrix;

/**
 * A class implementing Local Low-Rank Matrix Approximation.
 * Technical detail of the algorithm can be found in
 * Joonseok Lee and Seungyeon Kim and Guy Lebanon and Yoram Singer, Local Low-Rank Matrix Approximation,
 * Proceedings of the 30th International Conference on Machine Learning, 2013.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class SingletonGlobalLLORMA implements Recommender,ThetaRefresh {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	/** Rating matrix for items which will be used during the validation phase.
	 * (Deciding when to stop the gradient descent.)
	 * Do not directly use this during learning. */
	private SparseMatrix validationMatrix;
	/** Proportion of dataset, using for validation purpose. */
	private double validationRatio;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	double temp1, temp2, temp3;
	// Data structures:
	/** User profile in low-rank matrix form. */
	SparseMatrix[] localUserFeatures;
	/** Item profile in low-rank matrix form. */
	SparseMatrix[] localItemFeatures;
	/** Array of anchor user IDs. */
	int[] anchorUser;
	/** Array of anchor item IDs. */
	int[] anchorItem;
	/** Similarity matrix between users. */
	private static SparseMatrix userSimilarity;
	/** Similarity matrix between items. */
	private static SparseMatrix itemSimilarity;
	/** A global SVD model used for calculating user/item similarity. */
	public static RegularizedSVD baseline;
	/** Precalculated sum of weights for each user-item pair. */
	double[][] weightSum;
	
	// Local model parameters:
	/** The number of features. */
	public int featureCount;
	/** Learning rate parameter. */
	public double learningRate;
	/** Regularization factor parameter. */
	public double regularizer;
	/** Maximum number of iteration. */
	public int maxIter;
	
	// Global combination parameters:
	/** Maximum number of local models. */
	public int modelMax;
	/** Type of kernel used in kernel smoothing. */
	public int kernelType;
	/** Width of kernel used in kernel smoothing. */
	public double kernelWidth;
	
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress;
	
	
	public SparseMatrix testMatrix;
	
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param iter The maximum number of iterations.
	 * @param mm The maximum number of local models.
	 * @param kt Type of kernel used in kernel smoothing.
	 * @param kw Width of kernel used in kernel smoothing.
	 * @param vr The ratio of training set which will be used for validation purpose.
	 * @param base A global SVD model used for calculating user/item similarity.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 */
	public SingletonGlobalLLORMA(int uc, int ic, double max, double min, int fc,
			double lr, double r, int iter, int mm, int kt, double kw, double vr,
			RegularizedSVD base, SparseMatrix tm, boolean verbose) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		
		featureCount = fc;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		
		modelMax = mm;
		kernelType = kt;
		kernelWidth = kw;
		validationRatio = vr;
		baseline = base;
		
		testMatrix = tm;
		
		showProgress = verbose;
	}
	
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rateMatrix The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rateMatrix) {
//		makeValidationSet(rateMatrix, validationRatio);
		
		// Preparing data structures:
		localUserFeatures = new SparseMatrix[modelMax];
		localItemFeatures = new SparseMatrix[modelMax];
		
		anchorUser = new int[modelMax];
		anchorItem = new int[modelMax];
		
		for (int l = 0; l < modelMax; l++) {
			boolean done = false;
			while (!done) {
				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
				int[] itemList = rateMatrix.getRow(u_t).indexList();
	
				if (itemList != null) {
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					anchorUser[l] = u_t;
					anchorItem[l] = i_t;
					
					done = true;
				}
			}
		}
		
		// Pre-calculating similarity:
		userSimilarity = new SparseMatrix(userCount+1, userCount+1);
		itemSimilarity = new SparseMatrix(itemCount+1, itemCount+1);
		
		weightSum = new double[userCount+1][itemCount+1];
		for (int u = 1; u <= userCount; u++) {
			for (int i = 1; i <= itemCount; i++) {
				for (int l = 0; l < modelMax; l++) {
					weightSum[u][i] += KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType);
				}
			}
		}
		
		// Initialize local models:
		for (int l = 0; l < modelMax; l++) {
			localUserFeatures[l] = new SparseMatrix(userCount+1, featureCount);
			localItemFeatures[l] = new SparseMatrix(featureCount, itemCount+1);
			
			for (int u = 1; u <= userCount; u++) {
				for (int r = 0; r < featureCount; r++) {
					double rdm = Math.random();
					localUserFeatures[l].setValue(u, r, rdm);
				}
			}
			for (int i = 1; i <= itemCount; i++) {
				for (int r = 0; r < featureCount; r++) {
					double rdm = Math.random();
					localItemFeatures[l].setValue(r, i, rdm);
				}
			}
		}
		
		// Learn by gradient descent:
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		while (prevErr - currErr > 0.0001 && round < maxIter) {
			EvaluationMetrics e = this.evaluate(testMatrix);
			prevErr = currErr;
			currErr = e.getRMSE();
			
			// Show progress:
			if (showProgress) {
				
				System.out.println(round + "\t"	+ e.printOneLine());
			}
			System.out.println("theta in "+round+" is "+theta[0]);
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						// current estimation:
						double RuiEst = 0.0;
						for (int l = 0; l < modelMax; l++) {
							
//							double s_t_u = 0.0;
//							double Ku = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType);
//							double Ki = KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType);
//							s_t_u = Ku * Ki/ weightSum[u][i];
//							temp3 = 0.0;
//							for (int r = 0;r < featureCount; r++){
//								temp1 = 0.0;
//								temp2 = 0.0;
//								for (int k0 = 0; k0 < featureCount; k0++) {
//									temp1 += Math.exp(localUserFeatures[l].getValue(u ,k0));
//									if(k0 != r){																			
//										temp2 += s_t_u * localItemFeatures[l].getValue(k0, i);
//									}
//								}
//								if(temp1==0) {
//									temp1 = 1;
//									System.out.println("t0");
//								}
//								temp3 += Math.exp(localUserFeatures[l].getValue(u,r))/temp1
//										*(Math.exp(theta[0])*s_t_u*localItemFeatures[l].getValue(r, i)+temp2);																
//							}
//							RuiEst += temp3;
							
							
							RuiEst += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
									* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
								 	/ weightSum[u][i];
						}
						double RuiReal = rateMatrix.getValue(u, i);
						double err = RuiReal - RuiEst;
						
						// parameter update:
						for (int l = 0; l < modelMax; l++) {
							double weight = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 	  * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
									 	  / weightSum[u][i];
							
							for (int r = 0; r < featureCount; r++) {
								double Fus = localUserFeatures[l].getValue(u, r);
								double Gis = localItemFeatures[l].getValue(r, i);
								double Fus_t = 0.0,Gis_t = 0.0;
								double temp4 = 0.0,temp5 = 0.0;
								double temp6 = 0.0,temp7 = 0.0;
								for(int k=0;k<featureCount;k++){
									double Uuk0 = localUserFeatures[l].getValue(u, k);
									double Vik0 = localItemFeatures[l].getValue(k, i);
									temp4 += Math.exp(Uuk0);
									temp6 += Vik0;
									if(k!=r) {
										temp5 += Math.exp(Uuk0);
										temp7 += Vik0;
									}
								}	
								temp6 = Math.exp(Fus)/temp4-Math.pow(Math.exp(Fus), 2)/Math.pow(temp4, 2);
								Gis_t = (Gis* Math.exp(theta[0])+temp7)*temp6;
								//Gis = (Gis*theta[0]+temp7)/temp6;
								Fus_t = (Math.exp(Fus)* Math.exp(theta[0])+temp5)/temp4;
															
								localUserFeatures[l].setValue(u, r, Fus + learningRate*(err*Gis_t*weight - regularizer*Fus));
								if(Double.isNaN(Fus + learningRate*(err*Gis_t*weight - regularizer*Fus))) {
									System.out.println("a");
								}
								localItemFeatures[l].setValue(r, i, Gis + learningRate*(err*Fus_t*weight - regularizer*Gis));
								if(Double.isNaN(Gis + learningRate*(err*Fus_t*weight - regularizer*Gis))) {
									System.out.println("b");
								}
								double thetaSum = 0.0;
								
								for(int k=0;k<featureCount;k++){
									double Uuk0 = localUserFeatures[l].getValue(u, k);
									double Vik0 = localItemFeatures[l].getValue(k, i);								
									thetaSum += Math.exp(Uuk0)*Vik0*weight/temp4*Math.exp(theta[0]);		
								}
								// Update theta profile:
								if (thetaSum != 0) {
									//theta[0] -= learningRate * (thetaSum/ (double) userCount + 2*regularizer*theta[0] );
									theta[0] -= learningRate * thetaSum;
									if(Double.isInfinite(theta[0])) 
										theta[0] = 1.0;	
									else if(theta[0] < 0 || Double.isNaN(theta[0]))
										theta[0] = 0.001;
								}
							}
						}
					}
				}
			}
			
			round++;
		}
		
//		restoreValidationSet(rateMatrix);
	}
	
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector items = testMatrix.getRowRef(u);
			int[] itemIndexList = items.indexList();
			
			if (itemIndexList != null) {
				for (int i : itemIndexList) {
					double prediction = 0.0;
					for (int l = 0; l < modelMax; l++) {
						prediction += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
								* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
							 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
							 	/ weightSum[u][i];
					}
					
					if (Double.isNaN(prediction) || prediction == 0.0) {
						prediction = (maxValue + minValue)/2;
					}
					
					if (prediction < minValue) {
						prediction = minValue;
					}
					else if (prediction > maxValue) {
						prediction = maxValue;
					}
					
					predicted.setValue(u, i, prediction);
				}
			}
		}
		
		try {
			writeTxt(predicted,"CiaoDVD_pred_LLORMA_NCR");
			//writeTxt(predicted,"FilmTrust_pred_LLORMA_NCR");
			//writeTxt(predicted,"MovieLens_pred_LLORMA_NCR");
		} catch (IOException e) {
			// TODO Auto-generated catch block
				e.printStackTrace();
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	public void writeTxt(SparseMatrix sm,String name) throws IOException{
		String path = "LLORMA_data\\"   + name + ".txt";
		try {
			FileOutputStream out = new FileOutputStream(path);
			OutputStreamWriter outWriter = new OutputStreamWriter(out);
			BufferedWriter bufWrite = new BufferedWriter(outWriter);
			for(int u=0;u<=userCount;u++){
				bufWrite.write(sm.getRow(u)+"\r\n");
			}
			bufWrite.close();
			outWriter.close();
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Items which will be used for validation purpose are moved from rateMatrix to validationMatrix.
	 * 
	 * @param validationRatio Proportion of dataset, using for validation purpose.
	 */
	private void makeValidationSet(SparseMatrix rateMatrix, double validationRatio) {
		validationMatrix = new SparseMatrix(userCount+1, itemCount+1);
		
		int validationCount = (int) (rateMatrix.itemCount() * validationRatio);
		while (validationCount > 0) {
			int index = (int) (Math.random() * userCount) + 1;
			SparseVector row = rateMatrix.getRowRef(index);
			int[] itemList = row.indexList();
			
			if (itemList != null && itemList.length > 5) {
				int index2 = (int) (Math.random() * itemList.length);
				validationMatrix.setValue(index, itemList[index2], rateMatrix.getValue(index, itemList[index2]));
				rateMatrix.setValue(index, itemList[index2], 0.0);
				
				validationCount--;
			}
		}
	}
	
	/** Items in validationMatrix are moved to original rateMatrix. */
	private void restoreValidationSet(SparseMatrix rateMatrix) {
		for (int i = 1; i <= userCount; i++) {
			SparseVector row = validationMatrix.getRowRef(i);
			int[] itemList = row.indexList();
			
			if (itemList != null) {
				for (int j : itemList) {
					rateMatrix.setValue(i, j, validationMatrix.getValue(i, j));
				}
			}
		}
	}
	
	/**
	 * Calculate similarity between two users, based on the global base SVD.
	 * 
	 * @param idx1 The first user's ID.
	 * @param idx2 The second user's ID.
	 * 
	 * @return The similarity value between two users idx1 and idx2.
	 */
	private static double getUserSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = userSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = userSimilarity.getValue(idx2, idx1);
		}
		
		if (sim == 0.0) {
			SparseVector u_vec = baseline.getU().getRowRef(idx1);
			SparseVector v_vec = baseline.getU().getRowRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				userSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				userSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
	
	/**
	 * Calculate similarity between two items, based on the global base SVD.
	 * 
	 * @param idx1 The first item's ID.
	 * @param idx2 The second item's ID.
	 * 
	 * @return The similarity value between two items idx1 and idx2.
	 */
	private static double getItemSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = itemSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = itemSimilarity.getValue(idx2, idx1);
		} 
		
		if (sim == 0.0) {
			SparseVector i_vec = baseline.getV().getColRef(idx1);
			SparseVector j_vec = baseline.getV().getColRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				itemSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				itemSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
}