package prea.recommender.llorma;

import prea.data.structure.SparseVector;
import prea.data.structure.SparseMatrix;
import prea.util.RankEvaluator;

/**
 * A class for updating each local model.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 26
 * @version 1.3
 */
public class PairedLLORMAUpdater extends Thread implements ThetaRefresh{
	private SparseMatrix localUserFeature;
	private SparseMatrix localItemFeature;
	private int rank;
	private int userCount;
	private SparseMatrix rateMatrix;
	private SparseMatrix currentPrediction;
	private double[][] weightSum;
	//private float[][] weightSum;
	private int[] s_u;
	private int lossCode;
	private double learningRate;
	private double regularizer;
	private SparseVector w;
	private SparseVector v;
	private double temp1 ,temp2, temp3;
	
	public PairedLLORMAUpdater(SparseMatrix luf, SparseMatrix lif, SparseMatrix rm, SparseMatrix cp,
			double[][] ws, int[] su, int loss, double lr, double r, SparseVector w0, SparseVector v0) {
		localUserFeature = luf;
		localItemFeature = lif;
		userCount = (localUserFeature.length())[0];
		rank = (localUserFeature.length())[1];
		rateMatrix = rm;
		currentPrediction = cp;
		weightSum = ws;
		s_u = su;
		lossCode = loss;
		learningRate = lr;
		regularizer = r;
		w = w0;
		v = v0;
	}
	
	public SparseMatrix getUserFeature() {
		return localUserFeature;
	}
	
	public SparseMatrix getItemFeature() {
		return localItemFeature;
	}
	
	@Override
	public void run() {

		for (int u = 1; u <= userCount; u++) {
			int[] itemIndexList = rateMatrix.getRowRef(u).indexList();
			
			if (itemIndexList != null) {
				
				for (int r = 0; r < rank; r++) {
					double Uuk = localUserFeature.getValue(u, r);									
					double userSum = 0.0;
					//double thetaSum = 0.0;
					for (int i : itemIndexList) {
						double itemSum = 0.0;
						//double thetaSum = 0.0;
						double Vik = localItemFeature.getValue(r, i);
						double Mui = rateMatrix.getValue(u, i);
						double pred_i = currentPrediction.getValue(u, i);
						
						double Ku = w.getValue(u);
						double Ki = v.getValue(i);
						double s_t_u_i = Ku * Ki/ weightSum[u][i];
						for (int j : itemIndexList) {
							double pred_j = currentPrediction.getValue(u, j);
							double Muj = rateMatrix.getValue(u, j);
							double Vjk = localItemFeature.getValue(r, j);
							double Kj = v.getValue(j);
							double s_t_u_j = Ku * Kj/ weightSum[u][j];
	
							temp1 = 0.0;
							temp2 = 0.0;
							temp3 = 0.0;
							for (int k = 0; k < rank; k++){
								temp1 += Math.exp(localUserFeature.getValue(u, k));
								if(k != r){
									temp2 += s_t_u_i*localItemFeature.getValue(k, i)-s_t_u_j*localItemFeature.getValue(k, j);
									temp3 += Math.exp(localUserFeature.getValue(u, k));
									
								}
							}
														
							if (Mui > Muj) {
								double dg = RankEvaluator.lossDiff(Mui, Muj, pred_i, pred_j, lossCode);
														
								//userSum += (Vik*Ku*Ki/weightSum[u][i] - Vjk*Ku*Kj/weightSum[u][j]) * dg;
								//itemSum += Uuk*Ku*Ki/weightSum[u][i] * dg;
								userSum += (Math.exp(theta[0])*(s_t_u_i*Vik-s_t_u_j*Vjk)+temp2)*dg
										*(Math.exp(Uuk)/temp1-Math.pow(Math.exp(Uuk), 2)/Math.pow(temp1, 2));
								itemSum += (Uuk*Math.exp(theta[0])+temp3)*s_t_u_i*dg/temp1;
								
								if (Double.isInfinite(userSum)) {
									System.out.println("c");
								}
								if (Double.isInfinite(itemSum)) {
									System.out.println("d");
								}
							}
							
							else if (Mui < Muj) {
								double dg  = RankEvaluator.lossDiff(Muj, Mui, pred_j, pred_i, lossCode);
								//itemSum -= Uuk*Ku*Ki/weightSum[u][i] * dg;
								itemSum -= (Uuk*Math.exp(theta[0])+temp3)*s_t_u_j*dg/temp1;
							}
							//thetaSum += Math.exp(Uuk)*(Vik*s_t_u_i-Vjk*s_t_u_j)/temp1*Math.exp(theta[0]);
							
						}
						double thetaSum = 0.0;
						double temp4 =0.0;
						for(int k=0;k<rank;k++){
							double Uuk0 = localUserFeature.getValue(u, k);
							temp4+=Math.exp(Uuk0);
						}
						for(int k=0;k<rank;k++){
							double Uuk0 = localUserFeature.getValue(u, k);
							double Vik0 = localItemFeature.getValue(k, i);
							//double Vjk0 = localItemFeature.getValue(k, j);								
							thetaSum += Math.exp(Uuk0)*Vik0*s_t_u_i/temp4*Math.exp(theta[0]);		
							//thetaSum += Math.exp(Uuk0)*(Vik0*s_t_u_i-Vjk0*s_t_u_j)/temp1*Math.exp(theta[0]);
						}
						// Update theta profile:
						if (thetaSum != 0) {
							//theta[0] -= learningRate * (thetaSum/ (double) userCount / s_u[u]+ 2*regularizer*theta[0] );
							theta[0] -= 2*regularizer * thetaSum;
							if(Double.isInfinite(theta[0])) 
								theta[0] = 1.0;	
							else if(theta[0] < 0 || Double.isNaN(theta[0]))
								theta[0] = 0.001;
						}
						// Update item profiles:
						if (itemSum != 0) {
							localItemFeature.setValue(r, i, Vik - learningRate *(itemSum / (double) userCount / s_u[u] + 2*regularizer*Vik));
							if(Double.isNaN(Vik - learningRate *(itemSum / (double) userCount / s_u[u] + 2*regularizer*Vik))) {
								System.out.println("b1");
								localItemFeature.setValue(r, i, 0.001);
							}
							if(Double.isInfinite(Vik - learningRate *(itemSum / (double) userCount / s_u[u] + 2*regularizer*Vik))) {
								System.out.println("b2");
							}
						}
						//thetaSum += Math.exp(Uuk)*(Vik*s_t_u_i)/temp1*Math.exp(theta[0]);
						
					}
				
					// Update user profiles:
					if (userSum != 0) {
						localUserFeature.setValue(u, r, Uuk - learningRate *(userSum / (double) userCount / s_u[u] + 2*regularizer*Uuk));
						if(Double.isNaN(Uuk - learningRate *(userSum / (double) userCount / s_u[u] + 2*regularizer*Uuk))) {
							System.out.println("a1");
							localUserFeature.setValue(u, r, 0.001);
						}
						if(Double.isInfinite(Uuk - learningRate *(userSum / (double) userCount / s_u[u] + 2*regularizer*Uuk))) {
							System.out.println("a2");
							//localUserFeature.setValue(u, r, 0.99);
						}						
					}
					// Update theta profile:
//					if (thetaSum != 0) {
//						theta[0] -= learningRate * (thetaSum/ (double) userCount / s_u[u]+ 2*regularizer*theta[0] );					
//						if(Double.isInfinite(theta[0])) 
//							theta[0] = 0.1;	
//						else if(theta[0] < 0 || Double.isNaN(theta[0]))
//							theta[0] = 0.001;
//					}
				}
				
				
			}
		}
	}
}
