package prea.util;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This is a class providing rank-based evaluation metrics.
 * Unlike EvaluationMetrics class, this considers both (test, test)
 * and (test, train) pairs for evaluation purpose.
 * 
 * @author Joonseok Lee
 * @since 2013. 5. 28
 * @version 1.2
 */
public class RankEvaluator {
	private static final int NDCG_THRESHOLD = 10;
	
	// Loss code:
	public static final int LOSS_COUNT = 13;
	
	public static final int LOGISTIC_LOSS = 0;
	public static final int DISCRETE_LOSS = 1;
	public static final int LOG_LOSS_1 = 2;
	public static final int LOG_LOSS_2 = 3;
	public static final int EXP_LOSS_1 = 4;
	public static final int EXP_LOSS_2 = 5;
	public static final int HINGE_LOSS_1 = 6;
	public static final int HINGE_LOSS_2 = 7;
	public static final int ABSOLUTE_LOSS = 8;
	public static final int SQUARED_LOSS = 9;
	public static final int EXP_REGRESSION = 10;
	public static final int SMOOTH_L1_REGRESSION = 11;
	public static final int EXP_LOSS_new = 12;
	
	private SparseMatrix rateMatrix;
	private SparseMatrix testMatrix;
	private SparseMatrix U;
	private SparseMatrix V;
	private SparseMatrix predicted;
	
	private int userCount;
	private int itemCount;
	
	private double[] error;
	private double ndcg;
	private int activeUserCount;
	
	public RankEvaluator() {
		userCount = 0;
		itemCount = 0;
		
		error = new double[LOSS_COUNT];
		for (int i = 0; i < error.length; i++) {
			error[i] = 0.0;
		}
	}
	
	public RankEvaluator(SparseMatrix rm, SparseMatrix tm) {
		rateMatrix = rm;
		testMatrix = tm;
		
		userCount = rateMatrix.length()[0] - 1;
		itemCount = rateMatrix.length()[1] - 1;
		
		error = new double[LOSS_COUNT];
		for (int i = 0; i < error.length; i++) {
			error[i] = 0.0;
		}
	}
	
	public RankEvaluator(SparseMatrix rm, SparseMatrix tm, SparseMatrix p) {
		rateMatrix = rm;
		testMatrix = tm;
		predicted = p;
		
		userCount = rateMatrix.length()[0] - 1;
		itemCount = rateMatrix.length()[1] - 1;
		
		error = new double[LOSS_COUNT];
		for (int i = 0; i < error.length; i++) {
			error[i] = 0.0;
		}
		evaluate(true);
	}
	
//	public RankEvaluator(SparseMatrix rm, SparseMatrix tm, SparseMatrix u, SparseMatrix v) {
//		rateMatrix = rm;
//		testMatrix = tm;
//		U = u;
//		V = v;
//		
//		userCount = rateMatrix.length()[0] - 1;
//		
//		error = new double[LOSS_COUNT];
//		for (int i = 0; i < error.length; i++) {
//			error[i] = 0.0;
//		}
//		
//		evaluate(false);
//	}
	
	public void add(double[] userLoss) {
		for (int l = 0; l < LOSS_COUNT; l++) {
			error[l] += userLoss[l];
		}
		userCount++;
	}
	
	private void evaluate(boolean usePredicted) {
		activeUserCount = 0;
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector predictedItems = new SparseVector(itemCount+1);
			double[] userLoss = new double[LOSS_COUNT];
			
			int pairCount = 0;
			
			int[] trainItems = rateMatrix.getRowRef(u).indexList();
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				for (int i : testItems) {
					double realRate_i = testMatrix.getValue(u, i);
					double predictedRate_i = usePredicted ? predicted.getValue(u, i) : U.getRowRef(u).innerProduct(V.getColRef(i));
//					if (!usePredicted)
						predictedItems.setValue(i, predictedRate_i);
					
					for (int j : testItems) {
						double realRate_j = testMatrix.getValue(u, j);
						double predictedRate_j = usePredicted ? predicted.getValue(u, j) : U.getRowRef(u).innerProduct(V.getColRef(j));
						
						if (realRate_i > realRate_j) {
							for (int l = 0; l < LOSS_COUNT; l++) {
								userLoss[l] += loss(realRate_i, realRate_j, predictedRate_i, predictedRate_j, l);
							}
							pairCount++;
						}
					}
					
					if (trainItems != null) {
						for (int t = 0; t < trainItems.length; t++) {
							int j = trainItems[t];
							double realRate_j = rateMatrix.getValue(u, j);
							double predictedRate_j = usePredicted ? predicted.getValue(u, j) : U.getRowRef(u).innerProduct(V.getColRef(j));
//							if (!usePredicted && t == 0)
//								predictedItems.setValue(j, predictedRate_j);
							
							if (realRate_i > realRate_j) {
								for (int l = 0; l < LOSS_COUNT; l++) {
									userLoss[l] += loss(realRate_i, realRate_j, predictedRate_i, predictedRate_j, l);
								}
								pairCount++;
							}
						}
					}
				}
			}
			
			if (pairCount > 0) {
				for (int l = 0; l < LOSS_COUNT; l++) {
					userLoss[l] /= pairCount;
				}
			}
			
			for (int l = 0; l < LOSS_COUNT; l++) {
				error[l] += userLoss[l];
			}
			
/*			// Calculate NDCG:
			SparseVector observedItems = rateMatrix.getRowRef(u).plus(testMatrix.getRowRef(u));
			if (usePredicted) predictedItems = predicted.getRowRef(u);
*/			

			SparseVector observedItems = testMatrix.getRowRef(u);
			if (observedItems.itemCount() > 0) {
				int[] observedIndices = observedItems.indexList();
				double[] observedValues = observedItems.valueList();
				//int[] predictedIndices = predictedItems.indexList();
				double[] predictedValues = predictedItems.valueList();
				int listLength = Math.min(NDCG_THRESHOLD, observedIndices.length);

				double u_dcg = 0.0;
				Sort.kLargest(predictedValues, observedIndices, 0, observedIndices.length - 1, listLength);
				for (int i = 0; i < listLength; i++) {					
					double observed = rateMatrix.getValue(u, observedIndices[i]) + testMatrix.getValue(u, observedIndices[i]);
					u_dcg += (Math.pow(2.0, observed) - 1.0) / (Math.log(i+2) / Math.log(2.0));
				}
				
				double best_dcg = 0.0;
				Sort.kLargest(observedValues, observedIndices, 0, observedIndices.length - 1, listLength);
				// Now observedIndices were corrupted, but we do not need them.
				for (int i = 0; i < listLength; i++) {										
					best_dcg += (Math.pow(2.0, observedValues[i]) - 1.0) / (Math.log(i+2) / Math.log(2.0));
				}
				
				ndcg += (u_dcg / best_dcg);
				activeUserCount++;
			}
		}

		ndcg /= (double) activeUserCount;
	}
	
	public static double loss(double Mui, double Muj, double Fui, double Fuj, int lossCode) {
		switch (lossCode) {
		case LOGISTIC_LOSS:
			return 1 / (1 + Math.exp((Mui - Muj)*(Fui - Fuj)));
		case DISCRETE_LOSS:
			return (Mui - Muj)*(Fui - Fuj) < 0 ? 1 : 0;
		case LOG_LOSS_1:
			return (Mui - Muj) * Math.log(1 + Math.exp(Fuj - Fui));
		case LOG_LOSS_2:
			return Math.log(1 + Math.exp(Mui - Muj - Fui + Fuj));
		case EXP_LOSS_1:
			return (Mui - Muj) * Math.exp(Fuj - Fui);
		case EXP_LOSS_2:
			return Math.exp(Muj - Mui + Fuj - Fui);
		case HINGE_LOSS_1:
			return Math.max(Mui - Muj - Fui + Fuj, 0);
		case HINGE_LOSS_2:
			return (Mui - Muj) * Math.max(1 - Fui + Fuj, 0);
		case ABSOLUTE_LOSS:
			return Math.abs(Mui - Muj - Fui + Fuj);
		case SQUARED_LOSS:
			return Math.pow(Mui - Muj - Fui + Fuj, 2);
		case EXP_REGRESSION:
			return Math.exp(Mui - Muj - Fui + Fuj) + Math.exp(Fui - Fuj - Mui + Muj);
		case SMOOTH_L1_REGRESSION:
			return Math.log(1 + Math.exp(Mui - Muj - Fui + Fuj)) + Math.log(1 + Math.exp(Fui - Fuj - Mui + Muj));
		case EXP_LOSS_new:
			//return (Mui - Muj) *(Fuj - Fui)* Math.log(1 + Math.exp(Fuj - Fui));
			return (Mui - Muj) * Math.log(1 + Math.exp(Fuj - Fui));
		default:
			return 0.0;
		}
	}
	
	public static double lossDiff(double Mui, double Muj, double Fui, double Fuj, int lossCode) {
		switch (lossCode) {
		case LOGISTIC_LOSS:
			return (Muj - Mui) * Math.exp((Mui - Muj)*(Fui - Fuj)) / Math.pow(1 + Math.exp((Mui - Muj)*(Fui - Fuj)), 2);
		case DISCRETE_LOSS:
			return 0.0;
		case LOG_LOSS_1:
			return (Muj - Mui) / (1 + Math.exp(Fui - Fuj));
		case LOG_LOSS_2:
			return -1.0 / (1 + Math.exp(Fui - Fuj - Mui + Muj));
		case EXP_LOSS_1:
			return (Muj - Mui) * Math.exp(Fuj - Fui);
		case EXP_LOSS_2:
			return - Math.exp(Mui - Muj - Fui + Fuj);
		case HINGE_LOSS_1:
			return Mui - Muj > Fui - Fuj ? -1.0 : 0.0;
		case HINGE_LOSS_2:
			return Fui - Fuj < 1 ? Muj - Mui : 0.0;
		case ABSOLUTE_LOSS:
			if (Mui - Muj > Fui - Fuj)
				return -1.0;
			else if (Mui - Muj < Fui - Fuj)
				return 1.0;
			else
				return Math.random() * 2 - 1;
		case SQUARED_LOSS:
			return -2 * (Mui - Muj - Fui + Fuj);
		case EXP_REGRESSION:
			return Math.exp(Fui - Fuj - Mui + Muj) - Math.exp(Mui - Muj - Fui + Fuj);
		case SMOOTH_L1_REGRESSION:
			return -1.0 / (1 + Math.exp(Fui - Fuj - Mui + Muj)) + 1.0 / (1 + Math.exp(Mui - Muj - Fui + Fuj)); 
		case EXP_LOSS_new:
			return (Muj - Mui) * Math.exp(Fuj - Fui)/(1 + Math.exp(Fuj - Fui));
		default:
			return 0.0;
		}
	}
	
	public double getLoss(int lossCode) {
		if (lossCode >= 0 && lossCode < LOSS_COUNT) {
			return error[lossCode] / (double) activeUserCount;
		}
		else {
			return 0.0;
		}
	}
	
	public double getNDCG() {
		return ndcg;
	}
	
	
	public String printOneLine() {
		return String.format("%.4f\t%.4f\t",
			//String.format("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f",
			this.getLoss(EXP_LOSS_new),
			//this.getLoss(LOGISTIC_LOSS),
			//this.getLoss(LOG_LOSS_1),
			//this.getLoss(LOG_LOSS_2),
			//this.getLoss(EXP_LOSS_1),
			//this.getLoss(EXP_LOSS_2),
			//this.getLoss(HINGE_LOSS_1),
			//this.getLoss(HINGE_LOSS_2),
			//this.getLoss(ABSOLUTE_LOSS),
			//this.getLoss(SQUARED_LOSS),
			//this.getLoss(EXP_REGRESSION),
			//this.getLoss(SMOOTH_L1_REGRESSION),
			this.getNDCG()
			//this.getLoss(DISCRETE_LOSS)
		);
	}
	
	public static String printTitle() {
		return "=====================================================================================================================================================================\r\nName\tOptLoss\tRank\tExpn\tNDCG@" + NDCG_THRESHOLD + "\tMAP\tPre\tRec\tMRR\tRMSE\tMAE\tAUC";
		//return "=====================================================================================================================================================================\r\nName\tOptLoss\tRank\tExpn\tLogs\tLog1\tLog2\tExp1\tExp2\tHinge1\tHinge2\tAbs\tSqr\tExpReg\tSmL1\tNDCG@" + NDCG_THRESHOLD + "\t0/1";
	}
}
