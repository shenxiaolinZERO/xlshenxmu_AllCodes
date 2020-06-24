package prea.recommender.llorma;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.RankEvaluator;

/**
 * A class learning each local model used in paired LLORMA.
 * Implementation is based on weighted-SVD.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class WeakLearnerRank1 extends Thread {
	/** The unique identifier of the thread. */
	private int threadId;
	/** The number of features. */
	private int rank;
	/** The number of users. */
	private int userCount;
	/** The number of items. */
	private int itemCount;
	/** The anchor user used to learn this local model. */
	private int anchorUser;
	/** The anchor item used to learn this local model. */
	private int anchorItem;
	/** Learning rate parameter. */
	public double learningRate;
	/** The maximum number of iteration. */
	public int maxIter;
	/** Regularization factor parameter. */
	public double regularizer;
	/** The vector containing each user's weight. */
	private SparseVector w;
	/** The vector containing each item's weight. */
	private SparseVector v; 
	/** User profile in low-rank matrix form. */
	private SparseMatrix userFeatures;
	/** Item profile in low-rank matrix form. */
	private SparseMatrix itemFeatures;
	/** The rating matrix used for learning. */
	private SparseMatrix rateMatrix;
	/** The current train error. */
	private double trainErr;
	
	private static double[][] weightSum;
	private int[] s_u;
	private int lossCode;
	private boolean preset;
	
	private double lastError;
	private boolean continuing;
	
	/**
	 * Construct a local model for singleton LLORMA.
	 * 
	 * @param id A unique thread ID.
	 * @param rk The rank which will be used in this local model.
	 * @param u The number of users.
	 * @param i The number of items.
	 * @param au The anchor user used to learn this local model.
	 * @param ai The anchor item used to learn this local model.
	 * @param lr Learning rate parameter.
	 * @param r Regularization factor parameter.
	 * @param w0 Initial vector containing each user's weight.
	 * @param v0 Initial vector containing each item's weight.
	 * @param rm The rating matrix used for learning.
	 * @param loss The loss function to be minimized. See prea.util.RankEvaluator for loss codes.
	 * @param preU Initial low-rank user profile. (Use null for random assignment.)
	 * @param preV Initial low-rank item profile. (Use null for random assignment.)
	 */
	public WeakLearnerRank1(int id, int rk, int u, int i, int au, int ai, double lr, double r, 
			int iter, SparseVector w0, SparseVector v0, double[][] ws, SparseMatrix rm,
			int loss, SparseMatrix preU, SparseMatrix preV) {
		threadId = id;
		rank = rk;
		userCount = u;
		itemCount = i;
		anchorUser = au;
		anchorItem = ai;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		w = w0;
		v = v0;
		weightSum = ws;
		
		lastError = Double.MAX_VALUE;
		continuing = true;
		
		if (preU != null && preV != null) {
			preset = true;
			
			userFeatures = new SparseMatrix(userCount+1, rank);
			itemFeatures = new SparseMatrix(rank, itemCount+1);
			
			// Initialize user/item features:
			for (int a = 1; a <= userCount; a++) {
				for (int f = 0; f < rank; f++) {
					userFeatures.setValue(a, f, preU.getValue(a, f));
				}
			}
			for (int b = 1; b <= itemCount; b++) {
				for (int f = 0; f < rank; f++) {
					itemFeatures.setValue(f, b, preV.getValue(f, b));
				}
			}
		}
		else {
			userFeatures = new SparseMatrix(userCount+1, rank);
			itemFeatures = new SparseMatrix(rank, itemCount+1);
		}
		
		rateMatrix = rm;
		lossCode = loss;
	}
	
	/**
	 * Getter method for thread ID.
	 * 
	 * @return The thread ID of this local model.
	 */
	public int getThreadId() {
		return threadId;
	}
	
	/**
	 * Getter method for rank of this local model.
	 * 
	 * @return The rank of this local model.
	 */
	public int getRank() {
		return rank;
	}
	
	/**
	 * Getter method for anchor user of this local model.
	 * 
	 * @return The anchor user ID of this local model.
	 */
	public int getAnchorUser() {
		return anchorUser;
	}
	
	/**
	 * Getter method for anchor item of this local model.
	 * 
	 * @return The anchor item ID of this local model.
	 */
	public int getAnchorItem() {
		return anchorItem;
	}
	
	/**
	 * Getter method for user profile of this local model.
	 * 
	 * @return The user profile of this local model.
	 */
	public SparseMatrix getUserFeatures() {
		return userFeatures;
	}
	
	/**
	 * Getter method for item profile of this local model.
	 * 
	 * @return The item profile of this local model.
	 */
	public SparseMatrix getItemFeatures() {
		return itemFeatures;
	}
	
	/**
	 * Getter method for current train error.
	 * 
	 * @return The current train error.
	 */
	public double getTrainErr() {
		return trainErr;
	}
	
	/** Learn this local model based on similar users to the anchor user
	 * and similar items to the anchor item.
	 * Implemented with gradient descent. */
	@Override
	public void run() {
		//System.out.println("[START] Learning thread " + threadId);
		
		trainErr = Double.MAX_VALUE;
		boolean showProgress = false;
		
		if (!preset) {
			for (int u = 1; u <= userCount; u++) {
				for (int r = 0; r < rank; r++) {
					double rdm = Math.random() / rank;
					userFeatures.setValue(u, r, rdm);
				}
			}
			for (int i = 1; i <= itemCount; i++) {
				for (int r = 0; r < rank; r++) {
					double rdm = Math.random() / rank;
					itemFeatures.setValue(r, i, rdm);
				}
			}
		}
		
		s_u = new int[userCount+1];
		for (int u = 1; u <= userCount; u++) {
			s_u[u] = 0;
			int[] itemList = rateMatrix.getRowRef(u).indexList();
			if (itemList != null) {
				for (int i : itemList) {
					for (int j : itemList) {
						if (rateMatrix.getValue(u, i) > rateMatrix.getValue(u, j)) {
							s_u[u]++;
						}
					}
				}
			}
		}
		
		// Learn by Weighted RegSVD
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		while (continuing && round < maxIter) {
			double sum = 0.0;
			for (int u = 1; u <= userCount; u++) {
				int[] itemIndexList = rateMatrix.getRowRef(u).indexList();
				double sumTrain = 0.0;
				
				if (itemIndexList != null) {
					for (int r = 0; r < rank; r++) {
						double Uuk = userFeatures.getValue(u, r);
						double userSum = 0.0;
						double userCaseCount = s_u[u];
						
						for (int i : itemIndexList) {
							double itemSum = 0.0;
							double Vik = itemFeatures.getValue(r, i);
							double pred_i = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
							
							for (int j : itemIndexList) {
								double pred_j = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(j));
								double Mui = rateMatrix.getValue(u, i);
								double Muj = rateMatrix.getValue(u, j);
								double weight_i = w.getValue(u) * v.getValue(i) / weightSum[u][i];
								double weight_j = w.getValue(u) * v.getValue(j) / weightSum[u][j];
								
								if (Mui > Muj) {
									double dg = RankEvaluator.lossDiff(weight_i * Mui, weight_j * Muj, weight_i * pred_i, weight_j * pred_j, lossCode);
									double Vjk = itemFeatures.getValue(r, j);
									
									userSum += (weight_i * Vik - weight_j * Vjk) * dg;
									itemSum += weight_i * Uuk * dg;
									
									sumTrain += RankEvaluator.loss(weight_i * Mui, weight_j * Muj, weight_i * pred_i, weight_j * pred_j, lossCode);
								}
								
								else if (Mui < Muj) {
									double dg = RankEvaluator.lossDiff(weight_j * Muj, weight_i * Mui, weight_j * pred_j, weight_i * pred_i, lossCode);
									itemSum -= weight_i * Uuk * dg;
								}
							}
							
							// Update item profiles:
							if (itemSum != 0) {
								itemFeatures.setValue(r, i, Vik - learningRate/Math.sqrt(round+1) *((itemSum / (double) userCount / (double) userCaseCount) + 2*regularizer*Vik));
								if(Double.isNaN(Vik - learningRate/Math.sqrt(round+1) *((itemSum / (double) userCount / (double) userCaseCount) + 2*regularizer*Vik))) {
									// If you experience this, try to reduce learning rate.
									System.out.println("b");
								}
							}
						}
						
						if (userSum != 0) {
							userFeatures.setValue(u, r, Uuk - learningRate/Math.sqrt(round+1) *(userSum / (double) userCount / (double) userCaseCount + 2*regularizer*Uuk));
							if(Double.isNaN(Uuk - learningRate/Math.sqrt(round+1) *(userSum / (double) userCount / (double) userCaseCount + 2*regularizer*Uuk))) {
								// If you experience this, try to reduce learning rate.
								System.out.println("a");
							}
						}
						
						sumTrain /= (double) userCaseCount;
					}
					
					sumTrain /= (double) rank;
				}
				
				if (!Double.isNaN(sumTrain)) {
					sum += sumTrain;
				}
			}
			
			prevErr = currErr;
			currErr = sum/userCount;
			trainErr = Math.sqrt(currErr);
			
			round++;
			
			
			// Show progress:
			if (showProgress) {
				System.out.println(round + "\t" + currErr);
			}
		}
		
		//System.out.println("[END] Learning thread " + threadId);
	}
}