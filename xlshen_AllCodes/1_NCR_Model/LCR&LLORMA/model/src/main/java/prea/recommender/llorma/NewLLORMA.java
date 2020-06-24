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
 * THIS IS MORE EFFICIENT IMPLEMENTATION THAN ORIGINAL VERSION.
 * WE WILL REPLACE WITH THIS ONE LATER!
 * 
 * A class implementing Local Low-Rank Matrix Approximation.
 * Technical detail of the algorithm can be found in
 * Joonseok Lee and Seungyeon Kim and Guy Lebanon and Yoram Singer, Local Low-Rank Matrix Approximation,
 * Proceedings of the 30th International Conference on Machine Learning, 2013.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class NewLLORMA implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
//	public SparseMatrix rateMatrix;
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
	
	// Data structures:
	/** User profile in low-rank matrix form. */
	SparseMatrix[] localUserFeatures;
	/** Item profile in low-rank matrix form. */
	SparseMatrix[] localItemFeatures;
	/** Low-rank representation of anchor users. */
	SparseMatrix anchorUser;
	/** Low-rank representation of anchor items. */
	SparseMatrix anchorItem;
	/** Weight matrix between users. */
	private double[][] userWeight;
	/** Weight matrix between items. */
	private double[][] itemWeight;
	/** A global SVD model used for calculating user/item similarity. */
	public RegularizedSVD baseline;
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
	private int anchorOption;
	
	
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
	public NewLLORMA(int uc, int ic, double max, double min, int fc,
			double lr, double r, int iter, int mm, int kt, double kw, double vr,
			RegularizedSVD base, SparseMatrix tm, int ao, boolean verbose) {
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
		anchorOption = ao;
		
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
		
		if (anchorOption == 1) {
			initializeAnchorPointsRandom(rateMatrix);
		}
		else if (anchorOption == 2) {
			initializeAnchorPoints(rateMatrix);
		}
		else if (anchorOption == 4) {
			initializeAnchorPointsUniformRandom(rateMatrix);
		}
		else {
			initializeAnchorPointsByKmeans(rateMatrix);
		}
		
		
		// Initializing weights and weight sum:
		userWeight = new double[userCount+1][modelMax];
		itemWeight = new double[itemCount+1][modelMax];
		
		updateWeight();
		
		
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
		
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						// current estimation:
						double RuiEst = 0.0;
						for (int l = 0; l < modelMax; l++) {
							RuiEst += localUserFeatures[l].getRowRef(u).innerProduct(localItemFeatures[l].getColRef(i))
									* userWeight[u][l] * itemWeight[i][l] / weightSum[u][i];
						}
						double RuiReal = rateMatrix.getValue(u, i);
						double err = RuiReal - RuiEst;
						
						// parameter update:
						for (int l = 0; l < modelMax; l++) {
							double weight = userWeight[u][l] * itemWeight[i][l] / weightSum[u][i];
							
							for (int r = 0; r < featureCount; r++) {
								double Fus = localUserFeatures[l].getValue(u, r);
								double Gis = localItemFeatures[l].getValue(r, i);
								
								double userUpdate = learningRate*(err*Gis*weight - regularizer*Fus);
								if(!Double.isNaN(Fus + userUpdate)) {
									localUserFeatures[l].setValue(u, r, Fus + userUpdate);
								}
								else {
									if (weight > 0) {
										System.out.println("a");
									}
								}
								
								double itemUpdate = learningRate*(err*Fus*weight - regularizer*Gis);
								
								if(!Double.isNaN(Gis + itemUpdate)) {
									localItemFeatures[l].setValue(r, i, Gis + itemUpdate);
								}
								else {
									if (weight > 0) {
										System.out.println("b");
									}
								}
							}
						}
					}
				}
			}
			
//			// Anchor point update:
//			for (int u = 1; u <= userCount; u++) {
//				SparseVector items = rateMatrix.getRowRef(u);
//				int[] itemIndexList = items.indexList();
//				
//				if (itemIndexList != null) {
//					for (int i : itemIndexList) {
//						// current estimation:
//						double RuiEst = 0.0;
//						for (int l = 0; l < modelMax; l++) {
//							RuiEst += localUserFeatures[l].getRowRef(u).innerProduct(localItemFeatures[l].getColRef(i))
//									* userWeight[u][l] * itemWeight[i][l] / weightSum[u][i];
//						}
//						double RuiReal = rateMatrix.getValue(u, i);
//						
//						// parameter update:
//						for (int l = 0; l < modelMax; l++) {
//							// component 1
//							double err = RuiEst - RuiReal;
//							
//							// component 2
//							double common = (localUserFeatures[l].getRowRef(u).innerProduct(localItemFeatures[l].getColRef(i)) - RuiEst) / weightSum[u][i];
//							double u2 = common * itemWeight[i][l];
//							double i2 = common * userWeight[u][l];
//							
//							// component 3
//							SparseVector u_t = anchorUser.getRowRef(l);
//							SparseVector uu = baseline.getU().getRowRef(u);
//							double u_t_u = u_t.innerProduct(uu);
//							double u_t_norm = u_t.norm();
//							double uu_norm = uu.norm();
//							
//							SparseVector i_t = anchorItem.getRowRef(l);
//							SparseVector ii = baseline.getV().getColRef(i);
//							double i_t_i = i_t.innerProduct(ii);
//							double i_t_norm = i_t.norm();
//							double ii_norm = ii.norm();
//							
//							if (i_t.innerProduct(i_t) * ii.innerProduct(ii) < Math.pow(i_t_i, 2)) {
//								System.out.println("NAN");
//								i_t_i = i_t.innerProduct(ii);
//							}
//							
//							for (int f = 0; f < baseline.featureCount; f++) {
//								double u3 = 0.0;
//								double i3 = 0.0;
//								
//								if (u_t_u > 0 && u_t_u / (u_t_norm * uu_norm) < 0.9999) {
//									u3 = 4 / Math.PI * Math.acos(u_t_u / (u_t_norm * uu_norm))
//										* (uu.getValue(f) - u_t_u * u_t.getValue(f) / Math.pow(u_t_norm, 2))
//										/ Math.sqrt(u_t.innerProduct(u_t) * uu.innerProduct(uu) - Math.pow(u_t_u, 2));
//								}
//								if (i_t_i > 0 && i_t_i / (i_t_norm * ii_norm) < 0.9999) {
//									i3 = 4 / Math.PI * Math.acos(i_t_i / (i_t_norm * ii_norm))
//										* (ii.getValue(f) - i_t_i * i_t.getValue(f) / Math.pow(i_t_norm, 2))
//										/ Math.sqrt(i_t.innerProduct(i_t) * ii.innerProduct(ii) - Math.pow(i_t_i, 2));
//								}
//								
//								double u_update = err * u2 * u3 + 0*2*anchorUser.getValue(l, f);
//								anchorUser.setValue(l, f, anchorUser.getValue(l, f) - 0.0001 * u_update);
//								
//								double i_update = err * i2 * i3 + 0*2*anchorItem.getValue(l, f);
//								anchorItem.setValue(l, f, anchorItem.getValue(l, f) - 0.0001 * i_update);
//							}
//						}
//					}
//				}
//			}
//			
//			updateWeight();
			
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
								* userWeight[u][l] * itemWeight[i][l] / weightSum[u][i];
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
			writeTxt(predicted,"CiaoDVD_pred_LLORMA");
			//writeTxt(predicted,"FilmTrust_pred_LLORMA");
			//writeTxt(predicted,"MovieLens_pred_LLORMA");
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
	
	private double getUserWeight (int anchorIndex, int mainIndex) {
		return KernelSmoothing.kernelize(getUserSimilarity(anchorIndex, mainIndex), kernelWidth, kernelType);
	}
	
	private double getItemWeight (int anchorIndex, int mainIndex) {
		return KernelSmoothing.kernelize(getItemSimilarity(anchorIndex, mainIndex), kernelWidth, kernelType);
	}
	
	private double getUserSimilarity (int anchorIndex, int mainIndex) {
		SparseVector u_vec = anchorUser.getRowRef(anchorIndex);
		SparseVector v_vec = baseline.getU().getRowRef(mainIndex);
		
		double sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
		
		if (Double.isNaN(sim)) {
			sim = 0.0;
		}
		
		return sim;
	}
	
	private double getItemSimilarity (int anchorIndex, int mainIndex) {
		SparseVector i_vec = anchorItem.getRowRef(anchorIndex);
		SparseVector j_vec = baseline.getV().getColRef(mainIndex);
		
		double sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
		
		if (Double.isNaN(sim)) {
			sim = 0.0;
		}
		
		return sim;
	}
	
	private void updateWeight() {
		weightSum = new double[userCount+1][itemCount+1];
		for (int l = 0; l < modelMax; l++) {
			for (int u = 1; u <= userCount; u++) {
				double uw = getUserWeight(l, u);
				userWeight[u][l] = uw;
			}
			for (int i = 1; i <= itemCount; i++) {
				double iw = getItemWeight(l, i);
				itemWeight[i][l] = iw;
			}
		}
		
		for (int u = 1; u <= userCount; u++) {
			for (int i = 1; i <= itemCount; i++) {
				weightSum[u][i] = 0.0;
				for (int l = 0; l < modelMax; l++) {
					weightSum[u][i] += userWeight[u][l] * itemWeight[i][l];
				}
				
				if (weightSum[u][i] == 0.0) {
					weightSum[u][i] = modelMax;
					for (int l = 0; l < modelMax; l++) {
						userWeight[u][l] = 1.0;
						itemWeight[i][l] = 1.0;
					}
				}
			}
		}
	}
	
	// Method 1.
	private void initializeAnchorPointsRandom(SparseMatrix rateMatrix) {
		anchorUser = new SparseMatrix(modelMax, baseline.featureCount);
		anchorItem = new SparseMatrix(modelMax, baseline.featureCount);
		
		for (int l = 0; l < modelMax; l++) {
			int u_t = (int) Math.floor(Math.random() * userCount) + 1;
			int i_t = (int) Math.floor(Math.random() * itemCount) + 1;
			
			for (int f = 0; f < baseline.featureCount; f++) {
				anchorUser.setValue(l, f, baseline.getU().getValue(u_t, f));
				anchorItem.setValue(l, f, baseline.getV().getValue(f, i_t));
			}
		}
	}
	
	// Method 2.
	private void initializeAnchorPoints(SparseMatrix rateMatrix) {
		anchorUser = new SparseMatrix(modelMax, baseline.featureCount);
		anchorItem = new SparseMatrix(modelMax, baseline.featureCount);
		
		for (int l = 0; l < modelMax; l++) {
			boolean done = false;
			while (!done) {
				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
				int[] itemList = rateMatrix.getRowRef(u_t).indexList();
	
				if (itemList != null) {
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					for (int f = 0; f < baseline.featureCount; f++) {
						anchorUser.setValue(l, f, baseline.getU().getValue(u_t, f));
						anchorItem.setValue(l, f, baseline.getV().getValue(f, i_t));
					}
					
					done = true;
				}
			}
		}
	}
	
	// Method 4.
	private void initializeAnchorPointsUniformRandom(SparseMatrix rateMatrix) {
		int SAMPLE_COUNT = 50;
		
		anchorUser = new SparseMatrix(modelMax, baseline.featureCount);
		anchorItem = new SparseMatrix(modelMax, baseline.featureCount);
		
		int[] anchorUserIndex = new int[modelMax];
		int[] anchorItemIndex = new int[modelMax];
		
		for (int l = 0; l < modelMax; l++) {
			double maxMinDist = Double.MIN_VALUE;
			
			for (int s = 0; s < SAMPLE_COUNT; s++) {
				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
				int i_t = (int) Math.floor(Math.random() * itemCount) + 1;
				
				double minDist = Double.MAX_VALUE;
				for (int ll = 0; ll < l; ll++) {
					double currDist = 1 - getUserSimilarity(l, u_t) * getItemSimilarity(l, i_t);
					
					if (currDist < minDist) {
						minDist = currDist;
					}
				}
				
				if (minDist > maxMinDist) {
					maxMinDist = minDist;
					anchorUserIndex[l] = u_t;
					anchorItemIndex[l] = i_t;
				}
			}
			
			for (int f = 0; f < baseline.featureCount; f++) {
				anchorUser.setValue(l, f, baseline.getU().getValue(anchorUserIndex[l], f));
				anchorItem.setValue(l, f, baseline.getV().getValue(f, anchorItemIndex[l]));
			}
		}
	}
	
	private void initializeAnchorPointsByKmeans(SparseMatrix rateMatrix) {
		anchorUser = new SparseMatrix(modelMax, baseline.featureCount);
		anchorItem = new SparseMatrix(modelMax, baseline.featureCount);
		
		for (int l = 0; l < modelMax; l++) {
			boolean done = false;
			while (!done) {
				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
				int[] itemList = rateMatrix.getRowRef(u_t).indexList();
	
				if (itemList != null) {
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					for (int f = 0; f < baseline.featureCount; f++) {
						anchorUser.setValue(l, f, baseline.getU().getValue(u_t, f));
						anchorItem.setValue(l, f, baseline.getV().getValue(f, i_t));
					}
					
					done = true;
				}
			}
		}
		
		int[] userCluster = new int[userCount+1];
		boolean changed = true;
		
		while(changed) {
			changed = false;
			
			// Step 1: reassignment
			for (int u = 1; u <= userCount; u++) {
				double minDist = Double.MAX_VALUE;
				int minCluster = -1;
				
				SparseVector userVector = baseline.getU().getRowRef(u);
				
				for (int l = 0; l < modelMax; l++) {
					double dist = anchorUser.getRowRef(l).minus(userVector).norm();
					if (dist < minDist) {
						minDist = dist;
						minCluster = l;
					}
				}

				if (minCluster != userCluster[u]) {
					userCluster[u] = minCluster;
					changed = true;
				}
			}
			
			// Step 2: centroid recalculation
			int[] dataCount = new int[modelMax];
			anchorUser = new SparseMatrix(modelMax, baseline.featureCount);
			
			for (int u = 1; u <= userCount; u++) {
				SparseVector userVector = baseline.getU().getRowRef(u);
				
				for (int f = 0; f < baseline.featureCount; f++) {
					anchorUser.setValue(userCluster[u], f, anchorUser.getValue(userCluster[u], f) + userVector.getValue(f));
				}
				
				dataCount[userCluster[u]] += 1;
			}
			
			for (int l = 0; l < modelMax; l++) {
				for (int f = 0; f < baseline.featureCount; f++) {
					anchorUser.setValue(l, f, anchorUser.getValue(l, f) / (double) dataCount[l]);
				}
			}
		}
		
		int[] itemCluster = new int[itemCount+1];
		changed = true;
		
		while(changed) {
			changed = false;
			
			// Step 1: reassignment
			for (int i = 1; i <= itemCount; i++) {
				double minDist = Double.MAX_VALUE;
				int minCluster = -1;
				
				SparseVector itemVector = baseline.getV().getColRef(i);
				
				for (int l = 0; l < modelMax; l++) {
					double dist = anchorItem.getRowRef(l).minus(itemVector).norm();
					if (dist < minDist) {
						minDist = dist;
						minCluster = l;
					}
				}

				if (minCluster != itemCluster[i]) {
					itemCluster[i] = minCluster;
					changed = true;
				}
			}
			
			// Step 2: centroid recalculation
			int[] dataCount = new int[modelMax];
			anchorItem = new SparseMatrix(modelMax, baseline.featureCount);
			
			for (int i = 1; i <= itemCount; i++) {
				SparseVector itemVector = baseline.getV().getColRef(i);
				
				for (int f = 0; f < baseline.featureCount; f++) {
					anchorItem.setValue(itemCluster[i], f, anchorItem.getValue(itemCluster[i], f) + itemVector.getValue(f));
				}
				
				dataCount[itemCluster[i]] += 1;
			}
			
			for (int l = 0; l < modelMax; l++) {
				for (int f = 0; f < baseline.featureCount; f++) {
					anchorItem.setValue(l, f, anchorItem.getValue(l, f) / (double) dataCount[l]);
				}
			}
		}
	}
}
