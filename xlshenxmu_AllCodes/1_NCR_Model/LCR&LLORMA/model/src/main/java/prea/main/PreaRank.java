package prea.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

import prea.data.splitter.*;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.*;
import prea.recommender.matrix.RegularizedSVD;
import prea.recommender.matrix.RankBasedSVD;
import prea.recommender.llorma.WeakLearnerRank1;
import prea.recommender.llorma.WeakLearnerRank2;
import prea.recommender.llorma.PairedGlobalLLORMA;
import prea.recommender.llorma.PairedGlobalLLORMA2;
import prea.recommender.llorma.ThetaRefresh;
import prea.util.KernelSmoothing;
import prea.util.Printer;
import prea.util.RankEvaluator;
import prea.util.EvaluationMetrics;

//import prea.recommender.llorma.PairedLLORMAUpdater;

/**
 * An extension of LLORMA to rank-based losses.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 26
 * @version 1.2
 */
public class PreaRank implements ThetaRefresh{
	/** The maximum number of threads which will run simultaneously. 
	 *  We recommend not to exceed physical limit of your machine. */
	public static final int MULTI_THREAD_LEVEL = 8;
	
	/*========================================
	 * Parameters
	 *========================================*/
	/** The name of data file used for test. */
	public static String dataFileName;
	/** Evaluation mode */
	public static int evaluationMode;
	/** Proportion of items which will be used for test purpose. */
	public static double testRatio;
	/** The name of predefined split data file. */
	public static String splitFileName;
	/** The number of folds when k-fold cross-validation is used. */
	public static int foldCount;
	/** The number of training items for each user. */
	public static int userTrainCount;
	/** The number of test items guaranteed for each user. */
	public static int minTestCount;
	/** Indicating whether to run all algorithms. */
	public static boolean runAllAlgorithms;
	/** The code for an algorithm which will run. */
	public static String algorithmCode;
	/** Parameter list for the algorithm to run. */
	public static String[] algorithmParameters;
	/** Indicating whether pre-calculating user similarity or not */
	public static boolean userSimilarityPrefetch = false;
	/** Indicating whether pre-calculating item similarity or not */
	public static boolean itemSimilarityPrefetch = false;
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public static SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training and validation phase. */
	public static SparseMatrix testMatrix;
	/** Average of ratings for each user. */
	public static SparseVector userRateAverage;
	/** Average of ratings for each item. */
	public static SparseVector itemRateAverage;
	/** The number of users. */
	public static int userCount;
	/** The number of items. */
	public static int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public static int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public static int minValue;
	/** The list of item names, provided with the dataset. */
	public static String[] columnName;
	
	private static SparseMatrix userSimilarity;
	private static SparseMatrix itemSimilarity;
	public static RegularizedSVD baseline;
	
	public final static int PEARSON_CORR = 101;
	public final static int VECTOR_COS = 102;
	public final static int ARC_COS = 103;
	public final static int RATING_MATRIX = 111;
	public final static int MATRIX_FACTORIZATION = 112;
	
	/**
	 * Test examples for every algorithm. Also includes parsing the given parameters.
	 * 
	 * @param argv The argument list. Each element is separated by an empty space.
	 * First element is the data file name, and second one is the algorithm name.
	 * Third and later includes parameters for the chosen algorithm.
	 * Please refer to our web site for detailed syntax.
	 * @throws InterruptedException 
	 * @throws IOException 
	 */
	public static void main(String argv[]) throws InterruptedException, IOException {
		// Set default setting first:
		//dataFileName = "movieLens_100K";
		dataFileName = "MovieLens";
		//dataFileName = "FilmTrust";
		//dataFileName = "CiaoDVD";
		//dataFileName = "Tmall_hybrid";
		//dataFileName = "Tmall_single";
		//dataFileName = "Yoochoose";
		evaluationMode = DataSplitManager.SIMPLE_SPLIT;
		splitFileName = dataFileName + "_split.txt";
		testRatio = 0.2;
		foldCount = 8;
		
		// Parsing the argument:
		if (argv.length > 1) {
			parseCommandLine(argv);
		}
		
		// Read input file:
		//readArff (dataFileName + ".arff");
		readCsv (dataFileName + ".csv");
		//readTxt (dataFileName + ".txt");
		// Train/test data split:
		switch (evaluationMode) {
			case DataSplitManager.SIMPLE_SPLIT:
				SimpleSplit2 sSplit = new SimpleSplit2(rateMatrix, testRatio, maxValue, minValue);
				System.out.println("Evaluation\tSimple Split (" + (1 - testRatio) + " train, " + testRatio + " test)");
				testMatrix = sSplit.getTestMatrix();
				userRateAverage = sSplit.getUserRateAverage();
				itemRateAverage = sSplit.getItemRateAverage();
				
				run();
				break;
			case DataSplitManager.PREDEFINED_SPLIT:
				PredefinedSplit pSplit = new PredefinedSplit(rateMatrix, splitFileName, maxValue, minValue);
				System.out.println("Evaluation\tPredefined Split (" + splitFileName + ")");
				testMatrix = pSplit.getTestMatrix();
				userRateAverage = pSplit.getUserRateAverage();
				itemRateAverage = pSplit.getItemRateAverage();
				
				run();
				break;
			case DataSplitManager.K_FOLD_CROSS_VALIDATION:
				KfoldCrossValidation kSplit = new KfoldCrossValidation(rateMatrix, foldCount, maxValue, minValue);
				System.out.println("Evaluation\t" + foldCount + "-fold Cross-validation");
				for (int k = 1; k <= foldCount; k++) {
					testMatrix = kSplit.getKthFold(k);
					userRateAverage = kSplit.getUserRateAverage();
					itemRateAverage = kSplit.getItemRateAverage();
					
					run();
				}
				break;
			case DataSplitManager.RANK_EXP_SPLIT:
				RankExpSplit rSplit = new RankExpSplit(rateMatrix, userTrainCount, minTestCount, maxValue, minValue);
				System.out.println("Evaluation\t" + "Ranking Experiment with N = " + userTrainCount);
				testMatrix = rSplit.getTestMatrix();
				userRateAverage = rSplit.getUserRateAverage();
				itemRateAverage = rSplit.getItemRateAverage();
				
				run();
				break;
		}
	}
	
	/** Run an/all algorithm with given data, based on the setting from command arguments. 
	 * @throws InterruptedException 
	 * @throws IOException */
	private static void run() throws InterruptedException, IOException {
		System.out.println(RankEvaluator.printTitle() + "\tTrain Time\tTest Time");
		
		// Prefetching user/item similarity:
		if (userSimilarityPrefetch) {
			userSimilarity = calculateUserSimilarity(MATRIX_FACTORIZATION, ARC_COS, 0);
		}
		else {
			userSimilarity = new SparseMatrix(userCount+1, userCount+1);
		}
		
		if (itemSimilarityPrefetch) {
			itemSimilarity = calculateItemSimilarity(MATRIX_FACTORIZATION, ARC_COS, 0);
		}
		else {
			itemSimilarity = new SparseMatrix(itemCount+1, itemCount+1);
		}
		
		// Regularized SVD
		int[] svdRank = {1};
		//int[] svdRank = {1,10};
		
		for (int r : svdRank) {
			double learningRate = 0.005;
			double regularizer = 0.1;
			int maxIter = 100;
			
			baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue,
				r, learningRate, regularizer, 0, maxIter, false);
			System.out.println("SVD\tFro\t" + r + "\t" + testRecommender("SVD", baseline));
		}
		
		// Rank-based SVD
		int[] rsvdRank = {5};
		for (int r : rsvdRank) {
			// multiply 6 to the learning speed for MovieLens 1M data
//			runRankBasedSVD(RankEvaluator.LOG_LOSS_1, r, 1700*6, true);
//			runRankBasedSVD(RankEvaluator.LOG_LOSS_2, r, 1700*6, true);
//			runRankBasedSVD(RankEvaluator.EXP_LOSS_1, r, 370*6, true);
////			runRankBasedSVD(RankEvaluator.EXP_LOSS_2, r, 25, true);
//			runRankBasedSVD(RankEvaluator.HINGE_LOSS_1, r, 1700*6, true);
////			runRankBasedSVD(RankEvaluator.HINGE_LOSS_2, r, 1700, true);
////			runRankBasedSVD(RankEvaluator.EXP_REGRESSION, r, 40, true);
////			runRankBasedSVD(RankEvaluator.LOGISTIC_LOSS, r, 6000, true);
		}
		
		// Paired Global LLORMA
		System.out.println("this is LCR");
		runParallelGlobalLLORMA2(RankEvaluator.LOG_LOSS_1, 5,  3,  2000, true);
		
		// MovieLens 100K
		System.out.println("this is NCR");
		theta[0] = 0.1;
		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 5,  3,  2000, true);
//		System.out.println("this is NCR");
//		theta[0] = 0.1;
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  2500, true);
//		System.out.println("this is NCR,lr = 2500,theta = 0.2");
//		theta[0] = 0.2;
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  2500, true);	
//		System.out.println("this is NCR,lr = 2500,theta = 0.1");
//		theta[0] = 0.1;
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  2500, true);	
//		System.out.println("this is NCR,lr = 2500,theta = 0.05");
//		theta[0] = 0.05;
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  2500, true);	
		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  600, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  800, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  2000, true);		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  4000, true);		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  6000, true);		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  8000, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_new, 50,  5,  800, true);
		//runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 50,  5,  8000, true);
		//runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 50,  5, 14500, true);

//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 10,  5,  2200, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 20,  5,  4500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 30,  5,  7000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 40,  5,  8500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 50,  5, 10000, true);
/*		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 10,   700, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 10,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 10,  6500, true);
//
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 10,   880, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 10,  4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 10,  8000, true);

		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 10,  2500, true);
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 10, 12000, true);
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 10, 20000, true);
		
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5,  5,  6000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5, 10,  2800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5, 15,  1700, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5, 20,  1500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10,  5, 12000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10, 10,  5600, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10, 15,  3400, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10, 20,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 15,  5, 18000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 15, 10,  8400, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 15, 15,  5100, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 15, 20,  4500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20,  5, 24000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20, 10, 11200, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20, 15,  6800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20, 20,  6000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 30,  5, 36000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 30, 10, 17800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 30, 15, 10200, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 30, 20,  9000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40,  5, 48000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 10, 22400, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 15, 13600, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 40, 20, 12000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 50,  5, 60000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 50, 10, 26000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 50, 15, 17000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 50, 20, 15000, true);
		
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2,  5,  5,  1700, true);
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2,  5, 10,  1400, true);
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2,  5, 15,  1000, true);
		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2,  5, 20,   700, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 10,  5,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 10, 10,  2000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 10, 15,  1200, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 10, 20,   700, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 15,  5,  5000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 15, 10,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 15, 15,  1800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 15, 20,  1050, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 20,  5,  7000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 20, 10,  4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 20, 15,  2400, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 20, 20,  1400, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 30,  5, 10000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 30, 10,  6000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 30, 15,  3600, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 30, 20,  2100, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40,  5, 14000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 10,  8000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 15,  4800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 40, 20,  2800, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 50,  5, 18000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 50, 10, 10000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 50, 15,  6000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_2, 50, 20,  3500, true);
		
		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1,  5,  5,   500, true);
		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1,  5, 10,   250, true);
		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1,  5, 15,   150, true);
		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1,  5, 20,   100, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 10,  5,   450, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 10, 10,   220, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 10, 15,   140, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 10, 20,    90, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 20,  5,   900, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 20, 10,   440, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 20, 15,   280, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 20, 20,   180, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 30,  5,  1350, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 30, 10,   660, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 30, 15,   420, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 30, 20,   270, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40,  5,  1800, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 10,   880, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 15,   480, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 40, 20,   360, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 50,  5,  2150, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 50, 10,  1100, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 50, 15,   600, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_1, 50, 20,   450, true);
//		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2,  5,  1, 20, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2,  5,  5, 13, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2,  5, 10,  7, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2, 10,  1, 40, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2, 10,  5, 25, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_LOSS_2, 10, 10, 15, true);
		
		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1,  5,  5,  2500, true);
		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1,  5, 10,  1000, true);
		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1,  5, 15,   700, true);
		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1,  5, 20,   500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 10,  5,  2500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 10, 10,  1000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 10, 15,   600, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 10, 20,   400, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 15,  5,  4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 15, 10,  1500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 15, 15,   900, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 15, 20,   600, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 20,  5,  5500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 20, 10,  2000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 20, 15,  1200, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 20, 20,   800, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 30,  5,  9000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 30, 10,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 30, 15,  1800, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 30, 20,  1200, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 40,  5, 12000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 40, 10,  4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 40, 15,  2400, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 40, 20,  1600, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 50,  5, 15000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 50, 10,  5000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 50, 15,  3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_1, 50, 20,  2000, true);

//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5,  1, 1000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5,  5,  500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5, 10,  300, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5, 15,  200, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10,  1, 2000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10,  5, 1000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10, 10,  500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10, 15,  350, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 15,  1, 3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 15,  5, 1500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 15, 10,  800, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 15, 15,  600, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20,  1, 4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20,  5, 2000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20, 10, 1100, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20, 15,  800, true);
		
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION,  5,  1,  60, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION,  5,  5,  40, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION,  5, 10,  20, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION, 10,  1, 120, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION, 10,  5,  80, true);
//		runParallelGlobalLLORMA(RankEvaluator.EXP_REGRESSION, 10, 10,  40, true);
		
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5,  1, 4500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5,  5, 3500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5, 10, 2500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5, 15, 1500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10,  1, 9000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10,  5, 7000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10, 10, 5000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10, 15, 3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 15,  1, 13500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 15,  5, 10500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 15, 10, 7500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 15, 15, 4500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20,  1, 18000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20,  5, 14000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20, 10, 10000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20, 15, 6000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 30,  1, 27000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 30,  5, 21000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 30, 10, 15000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 30, 15, 9000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 40,  5, 28000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 40, 10, 20000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 40, 15, 12000, true);
*/		
		
		// MovieLens 1M
////		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5,  1, 9000, true);
////		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5,  5, 8000, true);
////		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS,  5, 10, 7500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10,  1, 18000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10,  5, 14000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 10, 10, 12000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20,  1, 30000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20,  5, 25000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOGISTIC_LOSS, 20, 10, 20000, true);
//		
////		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5,  1, 2500, true);
////		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5,  5, 2000, true);
////		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1,  5, 10, 1100, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10,  1, 5000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10,  5, 4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 10, 10, 2700, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20,  1, 9000, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20,  5, 7500, true);
//		runParallelGlobalLLORMA(RankEvaluator.LOG_LOSS_1, 20, 10, 5000, true);
//		
////		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5,  1, 1500, true);
////		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5,  5, 1200, true);
////		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2,  5, 10,  600, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10,  1, 2000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10,  5, 1500, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 10, 10, 1200, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20,  1, 4000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20,  5, 3000, true);
//		runParallelGlobalLLORMA(RankEvaluator.HINGE_LOSS_2, 20, 10, 2400, true);
		
		
		// Paired Parallel LLORMA
//		int maxIter = 20;
//		double kernelWidth = 0.8;
//		System.out.println(pairedParallelLLORMAOption2( 1, 5000, Math.min(testMatrix.itemCount(), 20), maxIter, EPANECHNIKOV_KERNEL, kernelWidth, RankEvaluator.LOG_LOSS_1, true));
//		System.out.println(pairedParallelLLORMAOption1( 5, 50000, Math.min(testMatrix.itemCount(), 48), maxIter, EPANECHNIKOV_KERNEL, kernelWidth, RankEvaluator.HINGE_LOSS_1, true));			
		
		
		//TensorSVD tensorSVD = new TensorSVD(userCount, itemCount, maxValue, minValue, 1, 0.0005, 0.1, 100, true, RankEvaluator.SQUARED_LOSS);
//		TensorSVD tensorSVD = new TensorSVD(userCount, itemCount, maxValue, minValue, 5, 0.02, 0.1, 10, true, RankEvaluator.LOG_LOSS_1, rateMatrix, testMatrix);
//		tensorSVD.buildModel(rateMatrix);
//		System.out.println(tensorSVD.evaluate(rateMatrix, testMatrix));
	}
	
	private static void runRankBasedSVD(int loss, int rank, double learningRate, boolean verbose) throws IOException {
		// Insensitive parameters are fixed with the following values:
		int maxIter = 40;
		double regularizer = 1E-6;
		
		RankBasedSVD rsvd = new RankBasedSVD(
			userCount, itemCount, maxValue, minValue,
			rank, learningRate, regularizer, 0, maxIter,
			loss, testMatrix, null, null, verbose);
			
		System.out.println("RSVD" + "\t" + loss + "\t" + rank + "\t" + testRecommender("RSVD", rsvd));
	}
	
	private static void runParallelGlobalLLORMA(int loss, int modelCount, int rank, double learningRate, boolean verbose) throws IOException {
		// Insensitive parameters are fixed with the following values:
		int maxIter = 5; //Init :200
		double kernelWidth = 0.8;
		int kernelType = KernelSmoothing.EPANECHNIKOV_KERNEL;
		double regularizer = 5E-9;

		PairedGlobalLLORMA pgllorma = new PairedGlobalLLORMA(
			userCount, itemCount, maxValue, minValue,
			rank, learningRate, regularizer, maxIter,
			Math.min(testMatrix.itemCount(), modelCount),
			kernelType, kernelWidth, loss, testMatrix, baseline,
			MULTI_THREAD_LEVEL, verbose);
		
		System.out.println("pgNCR" + modelCount + "\t" + loss + "\t" + rank + "\t" + testRecommender("pgLLR", pgllorma));
	}
	
	private static void runParallelGlobalLLORMA2(int loss, int modelCount, int rank, double learningRate, boolean verbose) throws IOException {
		// Insensitive parameters are fixed with the following values:
		int maxIter = 5; //Init :200
		double kernelWidth = 0.8;
		int kernelType = KernelSmoothing.EPANECHNIKOV_KERNEL;
		double regularizer = 5E-9;
		
		PairedGlobalLLORMA2 pgllorma = new PairedGlobalLLORMA2(
			userCount, itemCount, maxValue, minValue,
			rank, learningRate, regularizer, maxIter,
			Math.min(testMatrix.itemCount(), modelCount),
			kernelType, kernelWidth, loss, testMatrix, baseline,
			MULTI_THREAD_LEVEL, verbose);
		
		System.out.println("pgLLR" + modelCount + "\t" + loss + "\t" + rank + "\t" + testRecommender2("pgLLR", pgllorma));
	}
	
	/**
	 * Parse the command from user.
	 * 
	 * @param command The command string given by user.
	 */
	private static void parseCommandLine(String[] command) {
		int i = 0;
		
		while (i < command.length) {
			if (command[i].equals("-f")) { // input file
				dataFileName = command[i+1];
				i += 2;
			}
			else if (command[i].equals("-s")) { // data split
				if (command[i+1].equals("simple")) {
					evaluationMode = DataSplitManager.SIMPLE_SPLIT;
					testRatio = Double.parseDouble(command[i+2]);
				}
				else if (command[i+1].equals("pred")) {
					evaluationMode = DataSplitManager.PREDEFINED_SPLIT;
					splitFileName = command[i+2].trim();
				}
				else if (command[i+1].equals("kcv")) {
					evaluationMode = DataSplitManager.K_FOLD_CROSS_VALIDATION;
					foldCount = Integer.parseInt(command[i+2]);
				}
				else if (command[i+1].equals("rank")) {
					evaluationMode = DataSplitManager.RANK_EXP_SPLIT;
					userTrainCount = Integer.parseInt(command[i+2]);
					minTestCount = 10;
				}
				i += 3;
			}
			else if (command[i].equals("-a")) { // algorithm
				runAllAlgorithms = false;
				algorithmCode = command[i+1];
				
				// parameters for the algorithm:
				int j = 0;
				while (command.length > i+2+j && !command[i+2+j].startsWith("-")) {
					j++;
				}
				
				algorithmParameters = new String[j];
				System.arraycopy(command, i+2, algorithmParameters, 0, j);
				
				i += (j + 2);
			}
		}
	}
	
	/**
	 * Test interface for a rank-based recommender system.
	 * Print ranking-based measures for given test data.
	 * Note that we take (test-test) pairs as well as (train-test) pairs.
	 * No (train-train) pairs are used for testing.
	 * 
	 * @return evaluation metrics and elapsed time for learning and evaluation.
	 */
	public static String testRecommender(String algorithmName, Recommender r) throws IOException {
		long learnStart = System.currentTimeMillis();
		r.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics evalPointTrain = r.evaluate(rateMatrix);
		EvaluationMetrics evalPointTest = r.evaluate(testMatrix);
		RankEvaluator evalRank = new RankEvaluator(rateMatrix, testMatrix, evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()));
		long evalEnd = System.currentTimeMillis();
		try {
			writeTxt(rateMatrix,"_rate_NCR");
			writeTxt(testMatrix,"_test_NCR");
			writeTxt(evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()),"_pred_NCR");
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		//System.out.println(evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()).getRow(2));
		return evalRank.printOneLine() 
			 + String.format("%.4f", evalPointTest.getAveragePrecision()) + "\t"
			 + String.format("%.4f", evalPointTest.getPre()) + "\t"
			 + String.format("%.4f", evalPointTest.getRec()) + "\t"
			 + String.format("%.4f", evalPointTest.getMRR()) + "\t"
			 + String.format("%.4f", evalPointTest.getRMSE()) + "\t"
			 + String.format("%.4f", evalPointTest.getMAE()) + "\t"
			 + String.format("%.4f", evalPointTest.getAUC()) + "\t"
			 + Printer.printTime(learnEnd - learnStart) + "\t"
			 + Printer.printTime(evalEnd - evalStart);
		
	}
	
	/**
	 * Test interface for a rank-based recommender system.
	 * Print ranking-based measures for given test data.
	 * Note that we take (test-test) pairs as well as (train-test) pairs.
	 * No (train-train) pairs are used for testing.
	 * 
	 * @return evaluation metrics and elapsed time for learning and evaluation.
	 */
	public static String testRecommender2(String algorithmName, Recommender r) throws IOException {
		long learnStart = System.currentTimeMillis();
		r.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics evalPointTrain = r.evaluate(rateMatrix);
		EvaluationMetrics evalPointTest = r.evaluate(testMatrix);
		RankEvaluator evalRank = new RankEvaluator(rateMatrix, testMatrix, evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()));
		long evalEnd = System.currentTimeMillis();
		try {
			writeTxt(rateMatrix,"_rate_LCR");
			writeTxt(testMatrix,"_test_LCR");
			writeTxt(evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()),"_pred_LCR");
		}catch(FileNotFoundException e) {
			e.printStackTrace();
		}
		//System.out.println(evalPointTrain.getPrediction().plus(evalPointTest.getPrediction()).getRow(2));
		return evalRank.printOneLine() 
			 + String.format("%.4f", evalPointTest.getAveragePrecision()) + "\t"
			 + String.format("%.4f", evalPointTest.getPre()) + "\t"
			 + String.format("%.4f", evalPointTest.getRec()) + "\t"
			 + String.format("%.4f", evalPointTest.getMRR()) + "\t"
			 + String.format("%.4f", evalPointTest.getRMSE()) + "\t"
			 + String.format("%.4f", evalPointTest.getMAE()) + "\t"
			 + String.format("%.4f", evalPointTest.getAUC()) + "\t"
			 + Printer.printTime(learnEnd - learnStart) + "\t"
			 + Printer.printTime(evalEnd - evalStart);
		
	}
	
	public static void writeTxt(SparseMatrix sm,String name) throws IOException{
		String path = "dataMatrix\\" + dataFileName  + name + ".txt";
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
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the data file in ARFF format, and store it in rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 */
	private static void readArff(String fileName) {
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			ArrayList<String> tmpColumnName = new ArrayList<String>();
			
			String line;
			int userNo = 0; // sequence number of each user
			int attributeCount = 0;
			
			maxValue = -1;
			minValue = 99999;
			
			// Read attributes:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.contains("@ATTRIBUTE")) {
					String name;
					//String type;
					
					line = line.substring(10).trim();
					if (line.charAt(0) == '\'') {
						int idx = line.substring(1).indexOf('\'');
						name = line.substring(1, idx+1);
						//type = line.substring(idx+2).trim();
					}
					else {
						int idx = line.substring(1).indexOf(' ');
						name = line.substring(0, idx+1).trim();
						//type = line.substring(idx+2).trim();
					}
					
					//columnName[lineNo] = name;
					tmpColumnName.add(name);
					attributeCount++;
				}
				else if (line.contains("@RELATION")) {
					// do nothing
				}
				else if (line.contains("@DATA")) {
					// This is the end of attribute section!
					break;
				}
				else if (line.length() <= 0) {
					// do nothing
				}
			}
			
			// Set item count to data structures:
			itemCount = (attributeCount - 1)/2;
			columnName = new String[attributeCount];
			tmpColumnName.toArray(columnName);
			
			int[] itemRateCount = new int[itemCount+1];
			rateMatrix = new SparseMatrix(500000, itemCount+1); // Netflix: [480189, 17770]

			// Read data:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					line = line.substring(1, line.length() - 1);
					
					StringTokenizer st = new StringTokenizer (line, ",");
					
					while (st.hasMoreTokens()) {
						String token = st.nextToken().trim();
						int i = token.indexOf(" ");
						
						int movieID, rate;
						int index = Integer.parseInt(token.substring(0, i));
						String data = token.substring(i+1);
						
						if (index == 0) { // User ID
							//int userID = Integer.parseInt(data);
							
							userNo++;
						}
						else if (data.length() == 1) { // Rate
							movieID = index;
							rate = Integer.parseInt(data);
							
							if (rate > maxValue) {
								maxValue = rate;
							}
							else if (rate < minValue) {
								minValue = rate;
							}
							
							(itemRateCount[movieID])++;
							rateMatrix.setValue(userNo, movieID, rate);
						}
						else { // Date
							// Do not use
						}
					}
				}
			}
			
			userCount = userNo;
			
			// Reset user vector length:
			rateMatrix.setSize(userCount+1, itemCount+1);
			for (int i = 1; i <= itemCount; i++) {
				rateMatrix.getColRef(i).setLength(userCount+1);
			}
			
			System.out.println ("Data File\t" + dataFileName);
			System.out.println ("User Count\t" + userCount);
			System.out.println ("Item Count\t" + itemCount);
			System.out.println ("Rating Count\t" + rateMatrix.itemCount());
			System.out.println ("Rating Density\t" + String.format("%.2f", ((double) rateMatrix.itemCount() / (double) userCount / (double) itemCount * 100.0)) + "%");
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			System.exit(0);
		}
	}
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the data file in ARFF format, and store it in rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 */
	private static void readCsv(String fileName) {
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
					
			String line;
			int userNo = 943; // sequence number of each user
			//int userNo = 1235; //filmtrust
			//int userNo = 2664; //CiaoDVD
			int itemSum=0;
			
			String data[] = new String[7];
			maxValue = -1;
			minValue = 99999;

			rateMatrix = new SparseMatrix(50000, 18000); // Netflix: [480189, 17770]
			
			// Read data:
			while((line = buffer.readLine()) != null) {
				if (line.length() > 0) {
					data = line.split(",");	
					int userID,movieID1, movieID2;
					int rate,rate1,rate2;
					//double rate,rate1,rate2;
					userID = Integer.parseInt(data[0]);	
					movieID1 = Integer.parseInt(data[3]);	
					rate1 = Integer.parseInt(data[5]);
					//rate1 = Double.parseDouble(data[5]);
					rateMatrix.setValue(userID, movieID1, rate1);
					movieID2 = Integer.parseInt(data[4]);	
					rate2 = Integer.parseInt(data[6]);	
					//rate2 = Double.parseDouble(data[6]);
					rateMatrix.setValue(userID, movieID2, rate2);
					//int userID = Integer.parseInt(data[0]);
					if(rate1>rate2)	
						rate = rate1;
					else 	
						rate = rate2;
					if (rate > maxValue) {
						maxValue = (int) (rate);
					}
					else if (rate < minValue) {
						minValue = (int)(rate);
					}
					if(movieID1>movieID2){
						if(movieID1>itemSum)
							itemSum = movieID1;
					}
					else {
						if(movieID2>itemSum)
							itemSum = movieID2;
					}
						
				}
				else { // Date
						// Do not use
				}									
			}
			
			userCount = userNo;
			itemCount = itemSum;
			// Reset user vector length:
			rateMatrix.setSize(userCount+1, itemCount+1);
			for (int i = 1; i <= itemCount; i++) {
				rateMatrix.getColRef(i).setLength(userCount+1);
			}
			
			System.out.println ("Data File\t" + dataFileName);
			System.out.println ("User Count\t" + userCount);
			System.out.println ("Item Count\t" + itemCount);
			System.out.println ("Rating Count\t" + rateMatrix.itemCount());
			System.out.println ("Rating Density\t" + String.format("%.2f", ((double) rateMatrix.itemCount() / (double) userCount / (double) itemCount * 100.0)) + "%");
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			System.exit(0);
		}
	}
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the data file in ARFF format, and store it in rating matrix.
	 * Peripheral information such as max/min values, user/item count are also set in this method.
	 * 
	 * @param fileName The name of data file.
	 */
	private static void readTxt(String fileName) {
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
					
			String line;
			//int userNo = 62101; // sequence number of each user
			int userNo = 61376;//single
			//int userNo = 341391;//yoochoose
			int itemSum=0;
			
			String data[] = new String[7];
			maxValue = -1;
			minValue = 99999;

			rateMatrix = new SparseMatrix(65000, 200000); // Netflix: [480189, 17770]
			//rateMatrix = new SparseMatrix(350000, 31000); // Netflix: [480189, 17770]
			// Read data:
			while((line = buffer.readLine()) != null) {
				if (line.length() > 0) {
					data = line.split(",");	
					int userID,item, rate;

					userID = Integer.parseInt(data[0]);	
					item = Integer.parseInt(data[1]);
//					if(item>10)
//						item/=10;
					rate = Integer.parseInt(data[2]);
//					rate += 1;

					rateMatrix.setValue(userID, item, rate);
					//int userID = Integer.parseInt(data[0]);

					if (rate > maxValue) {
						maxValue = (int) (rate);
					}
					else if (rate < minValue) {
						minValue = (int)(rate);
					}
					if(item>itemSum){
						itemSum = item;
					}
						
				}
				else { // Date
						// Do not use
				}									
			}
			
			userCount = userNo;
			itemCount = itemSum;
			// Reset user vector length:
			rateMatrix.setSize(userCount+1, itemCount+1);
			for (int i = 1; i <= itemCount; i++) {
				rateMatrix.getColRef(i).setLength(userCount+1);
			}
			
			System.out.println ("Data File\t" + dataFileName);
			System.out.println ("User Count\t" + userCount);
			System.out.println ("Item Count\t" + itemCount);
			System.out.println ("Rating Count\t" + rateMatrix.itemCount());
			System.out.println ("Rating Density\t" + String.format("%.2f", ((double) rateMatrix.itemCount() / (double) userCount / (double) itemCount * 100.0)) + "%");
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			System.exit(0);
		}
	}
	
//	public static String pairedParallelLLORMAOption1(int rank, double learningRate, int modelMax, int maxIter, int kernelType, double kernelWidth, int lossCode, boolean verbose) throws InterruptedException {
//		final int multiThreadCount = 8;
//		
//		int completeModelCount = 0;
//		long learnStart = System.currentTimeMillis();
//		
//		int[] s_u = new int[userCount+1];
//		for (int u = 1; u <= userCount; u++) {
//			s_u[u] = 0;
//			int[] itemList = rateMatrix.getRowRef(u).indexList();
//			if (itemList != null) {
//				for (int i : itemList) {
//					for (int j : itemList) {
//						if (rateMatrix.getValue(u, i) > rateMatrix.getValue(u, j)) {
//							s_u[u]++;
//						}
//					}
//				}
//			}
//		}
//		
//		// Preparing anchor points:
//		WeakLearnerRank1[] learners = new WeakLearnerRank1[multiThreadCount];
//		int[] anchorUser = new int[modelMax];
//		int[] anchorItem = new int[modelMax];
//		
//		for (int l = 0; l < modelMax; l++) {
//			boolean done = false;
//			while (!done) {
//				int u_t = (int) Math.floor(Math.random() * userCount) + 1;
//				int[] itemList = rateMatrix.getRow(u_t).indexList();
//				
//				if (itemList != null) {
//					int idx = (int) Math.floor(Math.random() * itemList.length);
//					int i_t = itemList[idx];
//					
//					anchorUser[l] = u_t;
//					anchorItem[l] = i_t;
//					
//					done = true;
//				}
//			}
//		}
//		
//		// Pre-calculating similarity:
//		double[][] weightSum = new double[userCount+1][itemCount+1];
//		for (int u = 1; u <= userCount; u++) {
//			for (int i = 1; i <= itemCount; i++) {
//				for (int l = 0; l < modelMax; l++) {
//					weightSum[u][i] += (KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
//									  * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType));
//				}
//			}
//		}
//		
//		int modelCount = 0;
//		int[] runningThreadList = new int[multiThreadCount];
//		int runningThreadCount = 0;
//		int waitingThreadPointer = 0;
//		int nextRunningSlot = 0;
//		
//		double lowestError = Double.MAX_VALUE;
//		RankEvaluator preservedEval = null;
//		
//		boolean showProgress = verbose;
//		
//		// Testing option 1:
//		SparseMatrix cumPrediction = new SparseMatrix(userCount+1, itemCount+1);
//		SparseMatrix cumWeight = new SparseMatrix(userCount+1, itemCount+1);
//		
//		// Parallel training:
//		while (modelCount < modelMax || runningThreadCount > 0) {
//			int u_t = (int) Math.floor(Math.random() * userCount) + 1;
//			int[] itemList = rateMatrix.getRow(u_t).indexList();
//
//			if (itemList != null) {
//				if (runningThreadCount < multiThreadCount && modelCount < modelMax) {
//					// run a new thread:
//					int idx = (int) Math.floor(Math.random() * itemList.length);
//					int i_t = itemList[idx];
//					
//					anchorUser[modelCount] = u_t;
//					anchorItem[modelCount] = i_t;
//					
//					SparseVector w = kernelSmoothing(userCount+1, u_t, kernelType, kernelWidth, false);
//					SparseVector v = kernelSmoothing(itemCount+1, i_t, kernelType, kernelWidth, true);
//					
//					//RegularizedSVD preModel = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 10, 0.005, 0.25, 0, 100, false);
//					//System.out.println(testRecommender("Pre", preModel));
//					
//					learners[nextRunningSlot] = new WeakLearnerRank1(modelCount, rank, userCount, itemCount, u_t, i_t, learningRate, 1E-8, maxIter, w, v, weightSum, rateMatrix, lossCode, null, null);
//					learners[nextRunningSlot].start();
////System.out.println("Thread " + modelCount + " started.");
//					
//					runningThreadList[runningThreadCount] = modelCount;
//					runningThreadCount++;
//					modelCount++;
//					nextRunningSlot++;
//				}
//				else if (runningThreadCount > 0) {
//					learners[waitingThreadPointer].join();
////System.out.println("Thread " + waitingThreadPointer + " finished.");
//					
//					int mp = waitingThreadPointer;
//					int mc = completeModelCount;
//					completeModelCount++;
//					
//					//int testPairCount = 0;
//					
//					// Testing option 1:
//					SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
//					for (int u = 1; u <= userCount; u++) {
//						int[] trainItems = rateMatrix.getRowRef(u).indexList();
//						
//						if (trainItems != null) {
//							for (int i : trainItems) {
//								double weight = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType)
//											  * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType)
//											  / weightSum[u][i];
//								double newPrediction = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(i)) * weight;
//								cumWeight.setValue(u, i, cumWeight.getValue(u, i) + weight);
//								cumPrediction.setValue(u, i, cumPrediction.getValue(u, i) + newPrediction);
//								double prediction = cumPrediction.getValue(u, i);// / cumWeight.getValue(u, i);
//								
//								if (Double.isNaN(prediction)) {
//									prediction = 0.0;
//								}
//								
//								predicted.setValue(u, i, prediction);
//							}
//						}
//						
//						int[] testItems = testMatrix.getRowRef(u).indexList();
//						
//						if (testItems != null) {
//							for (int i : testItems) {
//								double weight = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType)
//											  * KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType)
//											  / weightSum[u][i];
//								double newPrediction = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(i)) * weight;
//								cumWeight.setValue(u, i, cumWeight.getValue(u, i) + weight);
//								cumPrediction.setValue(u, i, cumPrediction.getValue(u, i) + newPrediction);
//								double prediction = cumPrediction.getValue(u, i);// / cumWeight.getValue(u, i);
//								
//								if (Double.isNaN(prediction)) {
//									prediction = 0.0;
//								}
//								
//								predicted.setValue(u, i, prediction);
//							}
//						}
//					}
//					
//					
//					RankEvaluator eval = new RankEvaluator();
//					
//					for (int u = 1; u <= userCount; u++) {
//						double[] userRankLoss = new double[RankEvaluator.LOSS_COUNT];
//						for (int l = 0; l < userRankLoss.length; l++) {
//							userRankLoss[l] = 0.0;
//						}
//						
//						int userPairCount = 0;
//						
//						int[] trainItems = rateMatrix.getRowRef(u).indexList();
//						int[] testItems = testMatrix.getRowRef(u).indexList();
//						
//						if (testItems != null) {
//							for (int i : testItems) {
//								double realRate_i = testMatrix.getValue(u, i);
//								double predictedRate_i = predicted.getValue(u, i);
//								
//								for (int j : testItems) {
//									double realRate_j = testMatrix.getValue(u, j);
//									double predictedRate_j = predicted.getValue(u, j);
//									
//									if (realRate_i > realRate_j) {
//										for (int l = 0; l < userRankLoss.length; l++) {
//											userRankLoss[l] += RankEvaluator.loss(realRate_i, realRate_j, predictedRate_i, predictedRate_j, l);
//										}
//										
//										//testPairCount++;
//										userPairCount++;
//									}
//								}
//								
//								if (trainItems != null) {
//									for (int j : trainItems) {
//										double realRate_j = rateMatrix.getValue(u, j);
//										double predictedRate_j = predicted.getValue(u, i);
//										
//										if (realRate_i > realRate_j) {
//											for (int l = 0; l < userRankLoss.length; l++) {
//												userRankLoss[l] += RankEvaluator.loss(realRate_i, realRate_j, predictedRate_i, predictedRate_j, l);
//											}
//											
//											//testPairCount++;
//											userPairCount++;
//										}
//									}
//								}
//							}
//						}
//						
//						if (userPairCount > 0) {
//							for (int l = 0; l < userRankLoss.length; l++) {
//								userRankLoss[l] /= userPairCount;
//							}
//							eval.add(userRankLoss);
//						}
//					}
//					
//					if (showProgress) {
//						System.out.println((modelCount - multiThreadCount + 1) + "\t\t\t" + eval.printOneLine());
//					}
//					
//					if (eval.getLoss(lossCode) < lowestError) {
//						lowestError = eval.getLoss(lossCode);
//						preservedEval = eval;
//					}
//					
//					
//					nextRunningSlot = waitingThreadPointer;
//					waitingThreadPointer = (waitingThreadPointer + 1) % multiThreadCount;
//					runningThreadCount--;
//					
//					// Release memory used for similarity prefetch
//					int au = anchorUser[mc];
//					for (int u = 1; u <= userCount; u++) {
//						if (au <= u) {
//							userSimilarity.setValue(au, u, 0.0);
//						}
//						else {
//							userSimilarity.setValue(u, au, 0.0);
//						}
//					}
//					int ai = anchorItem[mc];
//					for (int i = 1; i <= itemCount; i++) {
//						if (ai <= i) {
//							itemSimilarity.setValue(ai, i, 0.0);
//						}
//						else {
//							itemSimilarity.setValue(i, ai, 0.0);
//						}
//					}
//				}
//			}
//		}
//		
//		long learnEnd = System.currentTimeMillis();
//		
//		return "LLORMA\t" + lossCode + "\t" + rank + "\t"
//			+ preservedEval.printOneLine() + "\t"
//			+ Printer.printTime(learnEnd - learnStart) + "\t"
//			+ Printer.printTime(0);
//	}
	
	public static String pairedParallelLLORMAOption2(int rank, double learningRate, int modelMax, int maxIter, int kernelType, double kernelWidth, int lossCode, boolean verbose) throws InterruptedException {
		final int RUNNING_THREAD_MAX = 1;
		
		int completeModelCount = 0;
		long learnStart = System.currentTimeMillis();
		
		WeakLearnerRank2[] learners = new WeakLearnerRank2[RUNNING_THREAD_MAX];
		int[] anchorUser = new int[modelMax];
		int[] anchorItem = new int[modelMax];
		
		int modelCount = 0;
		int[] runningThreadList = new int[RUNNING_THREAD_MAX];
		int runningThreadCount = 0;
		int waitingThreadPointer = 0;
		int nextRunningSlot = 0;
		
		double lowestError = Double.MAX_VALUE;
		RankEvaluator preservedEval = null;
		
		boolean showProgress = verbose;
		
		// Count test pairs:
		int pairBound = 1;
		for (int u = 1; u <= userCount; u++) {
			int[] trainItems = rateMatrix.getRowRef(u).indexList();
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				for (int i : testItems) {
					double realRate_i = testMatrix.getValue(u, i);
					
					for (int j : testItems) {
						double realRate_j = testMatrix.getValue(u, j);
						
						if (realRate_i > realRate_j) {
							pairBound++;
						}
					}
					
					if (trainItems != null) {
						for (int j : trainItems) {
							double realRate_j = rateMatrix.getValue(u, j);
							
							if (realRate_i > realRate_j) {
								pairBound++;
							}
						}
					}
				}
			}
		}
		
		// Testing option 2:
		double[] cumPrediction = new double[pairBound];
		double[] cumWeight = new double[pairBound];
		
		// Parallel training:
		while (modelCount < modelMax) {
			int u_t = (int) Math.floor(Math.random() * userCount) + 1;
			int[] itemList = rateMatrix.getRow(u_t).indexList();

			if (itemList != null) {
				if (runningThreadCount < RUNNING_THREAD_MAX) {
					// run a new thread:
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					anchorUser[modelCount] = u_t;
					anchorItem[modelCount] = i_t;
					
					SparseVector w = kernelSmoothing(userCount+1, u_t, kernelType, kernelWidth, false);
					SparseVector v = kernelSmoothing(itemCount+1, i_t, kernelType, kernelWidth, true);
					
					//RegularizedSVD preModel = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 10, 0.005, 0.25, 0, 100, false);
					//System.out.println(testRecommender("Pre", preModel));
					
					learners[nextRunningSlot] = new WeakLearnerRank2(modelCount, rank, userCount, itemCount, u_t, i_t, learningRate, 1E-6, maxIter, w, v, rateMatrix, lossCode, null, null);
					learners[nextRunningSlot].start();
//System.out.println("Thread " + modelCount + " started.");
					
					runningThreadList[runningThreadCount] = modelCount;
					runningThreadCount++;
					modelCount++;
					nextRunningSlot++;
				}
				else if (runningThreadCount > 0) {
					learners[waitingThreadPointer].join();
//System.out.println("Thread " + waitingThreadPointer + " finished.");
					
					int mp = waitingThreadPointer;
					int mc = completeModelCount;
					completeModelCount++;
					
					int testPairCount = 0;
					
					
					// Testing option 2:
					RankEvaluator eval = new RankEvaluator();
					
					for (int u = 1; u <= userCount; u++) {
						double[] userRankLoss = new double[RankEvaluator.LOSS_COUNT];
						for (int l = 0; l < userRankLoss.length; l++) {
							userRankLoss[l] = 0.0;
						}
						
						int userPairCount = 0;
						
						int[] trainItems = rateMatrix.getRowRef(u).indexList();
						int[] testItems = testMatrix.getRowRef(u).indexList();
						
						if (testItems != null) {
							for (int i : testItems) {
								double realRate_i = testMatrix.getValue(u, i);
								double predictedRate_i = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(i));
								
								for (int j : testItems) {
									double realRate_j = testMatrix.getValue(u, j);
									double predictedRate_j = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(j));
									double weight = WeakLearnerRank2.getWeight(
											KernelSmoothing.kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType),
											KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType),
											KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], j), kernelWidth, kernelType), 4);
									
									if (realRate_i > realRate_j) {
										cumPrediction[testPairCount] += (predictedRate_i - predictedRate_j) * weight;
										cumWeight[testPairCount] += weight;
										
										double prediction = cumPrediction[testPairCount] / cumWeight[testPairCount];
										if (Double.isNaN(prediction)) {
											prediction = 0.0;
										}
										
										for (int l = 0; l < userRankLoss.length; l++) {
											userRankLoss[l] += RankEvaluator.loss(realRate_i, realRate_j, prediction, 0.0, l);
										}
										
										testPairCount++;
										userPairCount++;
									}
								}
								
								if (trainItems != null) {
									for (int j : trainItems) {
										double realRate_j = rateMatrix.getValue(u, j);
										double predictedRate_j = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(j));
										double weight = WeakLearnerRank2.getWeight(
												KernelSmoothing.kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType),
												KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType),
												KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], j), kernelWidth, kernelType), 4);
										
										if (realRate_i > realRate_j) {
											cumPrediction[testPairCount] += (predictedRate_i - predictedRate_j) * weight;
											cumWeight[testPairCount] += weight;
											
											double prediction = cumPrediction[testPairCount] / cumWeight[testPairCount];
											if (Double.isNaN(prediction)) {
												prediction = 0.0;
											}
											
											for (int l = 0; l < userRankLoss.length; l++) {
												userRankLoss[l] += RankEvaluator.loss(realRate_i, realRate_j, prediction, 0.0, l);
											}
											
											testPairCount++;
											userPairCount++;
										}
									}
								}
							}
						}
						
						if (userPairCount > 0) {
							for (int l = 0; l < userRankLoss.length; l++) {
								userRankLoss[l] /= userPairCount;
							}
							eval.add(userRankLoss);
						}
					}

					
					if (showProgress) {
						System.out.println((modelCount - RUNNING_THREAD_MAX + 1) + "\t\t\t" + eval.printOneLine());
					}
					
					if (eval.getLoss(lossCode) < lowestError) {
						lowestError = eval.getLoss(lossCode);
						preservedEval = eval;
					}
					
					
					nextRunningSlot = waitingThreadPointer;
					waitingThreadPointer = (waitingThreadPointer + 1) % RUNNING_THREAD_MAX;
					runningThreadCount--;
					
					// Release memory used for similarity prefetch
					int au = anchorUser[mc];
					for (int u = 1; u <= userCount; u++) {
						if (au <= u) {
							userSimilarity.setValue(au, u, 0.0);
						}
						else {
							userSimilarity.setValue(u, au, 0.0);
						}
					}
					int ai = anchorItem[mc];
					for (int i = 1; i <= itemCount; i++) {
						if (ai <= i) {
							itemSimilarity.setValue(ai, i, 0.0);
						}
						else {
							itemSimilarity.setValue(i, ai, 0.0);
						}
					}
				}
			}
		}
		
		long learnEnd = System.currentTimeMillis();
		
		return "LLORMA\t" + lossCode + "\t" + rank + "\t"
			+ preservedEval.printOneLine() + "\t"
			+ Printer.printTime(learnEnd - learnStart) + "\t"
			+ Printer.printTime(0);
	}
	
/*
	public static String pairedGlobalLLORMA(int rank, double learningRate, int modelMax, int maxIter, int kernelType, double kernelWidth, int lossCode, boolean verbose) {
		SparseMatrix[] localUserFeatures = new SparseMatrix[modelMax];
		SparseMatrix[] localItemFeatures = new SparseMatrix[modelMax];
		EvaluationMetrics preservedEvalPoint = null;
		RankEvaluator preservedEvalRank = null;
		
		long learnStart = System.currentTimeMillis();
		
		int[] s_u = new int[userCount+1];
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
		
		// Preparing anchor points:
		int[] anchorUser = new int[modelMax];
		int[] anchorItem = new int[modelMax];
		
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
		double[][] weightSum = new double[userCount+1][itemCount+1];
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
			localUserFeatures[l] = new SparseMatrix(userCount+1, rank);
			localItemFeatures[l] = new SparseMatrix(rank, itemCount+1);
			
			for (int u = 1; u <= userCount; u++) {
				for (int r = 0; r < rank; r++) {
					double rdm = Math.random();
					localUserFeatures[l].setValue(u, r, rdm);
				}
			}
			for (int i = 1; i <= itemCount; i++) {
				for (int r = 0; r < rank; r++) {
					double rdm = Math.random();
					localItemFeatures[l].setValue(r, i, rdm);
				}
			}
		}
		
		// Learn by gradient descent:
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		double regularizer = 1E-8;
		boolean showProgress = true;
		
		while (round < maxIter) {
			SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1); // Current prediction
			
			// Point evaluation is only with test points:
			for (int uu = 1; uu <= userCount; uu++) {
				int[] testItems = testMatrix.getRowRef(uu).indexList();
				
				if (testItems != null) {
					for (int i : testItems) {
						double prediction = 0.0;
						for (int l = 0; l < modelMax; l++) {
							prediction += localUserFeatures[l].getRow(uu).innerProduct(localItemFeatures[l].getCol(i))
									* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], uu), kernelWidth, kernelType)
								 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
								 	/ weightSum[uu][i];
						}
						
						if (Double.isNaN(prediction)) {
							prediction = 0.0;
						}
						predicted.setValue(uu, i, prediction);
					}
				}
			}
			
			EvaluationMetrics evalPoint = new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
			
			// Rank-based evaluation takes (train-test) pairs as well as (test-test) points:
			for (int uu = 1; uu <= userCount; uu++) {
				int[] trainItems = rateMatrix.getRowRef(uu).indexList();
				
				if (trainItems != null) {
					for (int i : trainItems) {
						double prediction = 0.0;
						for (int l = 0; l < modelMax; l++) {
							prediction += localUserFeatures[l].getRow(uu).innerProduct(localItemFeatures[l].getCol(i))
									* KernelSmoothing.kernelize(getUserSimilarity(anchorUser[l], uu), kernelWidth, kernelType)
								 	* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
								 	/ weightSum[uu][i];
						}
						
						if (Double.isNaN(prediction)) {
							prediction = 0.0;
						}
						predicted.setValue(uu, i, prediction);
					}
				}
			}
			
			RankEvaluator evalRank = new RankEvaluator(rateMatrix, testMatrix, predicted);
			prevErr = currErr;
			currErr = evalRank.getLoss(lossCode);
			
			
			if (showProgress) {
				System.out.println(round + "\t" + lossCode + "\t" + rank + "\t" + evalRank.printOneLine() + "\t" + String.format("%.4f", evalPoint.getAveragePrecision()));
			}

			if (prevErr > currErr) {
				preservedEvalPoint = evalPoint;
				preservedEvalRank = evalRank;
			}
			else {
				break;
			}
			
			// Gradient Descent for each local model in parallel:
			int multiThreadCount = 8;
			
			int modelCount = 0;
			int completeModelCount = 0;
			int[] runningThreadList = new int[multiThreadCount];
			int runningThreadCount = 0;
			int waitingThreadPointer = 0;
			int nextRunningSlot = 0;
			
			//int l = 0;
			PairedLLORMAUpdater[] updater = new PairedLLORMAUpdater[multiThreadCount];
			
			while (modelCount < modelMax || runningThreadCount > 0) {
				if (runningThreadCount < multiThreadCount && modelCount < modelMax) {
					SparseVector w = kernelSmoothing(userCount+1, anchorUser[modelCount], kernelType, kernelWidth, false);
					SparseVector v = kernelSmoothing(itemCount+1, anchorItem[modelCount], kernelType, kernelWidth, true);
					
					updater[nextRunningSlot] = new PairedLLORMAUpdater(localUserFeatures[modelCount], localItemFeatures[modelCount], rateMatrix, predicted, weightSum, s_u, lossCode, learningRate, regularizer, w, v);
					updater[nextRunningSlot].start();
					
					runningThreadList[runningThreadCount] = modelCount;
					runningThreadCount++;
					modelCount++;
					nextRunningSlot++;
				}
				else if (runningThreadCount > 0) {
					try {
						updater[waitingThreadPointer].join();
						
						int mp = waitingThreadPointer;
						int mc = completeModelCount;
						completeModelCount++;
						
						nextRunningSlot = waitingThreadPointer;
						waitingThreadPointer = (waitingThreadPointer + 1) % multiThreadCount;
						runningThreadCount--;
					}
					catch(InterruptedException ie) {
						System.out.println("Join failed: " + ie);
					}
				}
			}
			
			round++;
		}
		
		long learnEnd = System.currentTimeMillis();
		
		return "pgLLR\t" + lossCode + "\t" + rank + "\t"
			+ preservedEvalRank.printOneLine() + "\t"
			+ String.format("%.4f", preservedEvalPoint.getAveragePrecision()) + "\t"
			+ Printer.printTime(learnEnd - learnStart) + "\t"
			+ Printer.printTime(0);
	}
*/
	
	public static double getUserSimilarity (int idx1, int idx2) {
		if (userSimilarityPrefetch) {
			return userSimilarity.getValue(idx1, idx2);
		}
		else {
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
	}
	
	public static double getItemSimilarity (int idx1, int idx2) {
		if (itemSimilarityPrefetch) {
			return itemSimilarity.getValue(idx1, idx2);
		}
		else {
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
	
	public static SparseVector kernelSmoothing(int size, int id, int kernelType, double width, boolean isItemFeature) {
		SparseVector newFeatureVector = new SparseVector(size);
		newFeatureVector.setValue(id, 1.0);
		
		for (int i = 1; i < size; i++) {
			double sim;
			if (isItemFeature) {
				sim = getItemSimilarity(i, id);
			}
			else { // userFeature
				sim = getUserSimilarity(i, id);
			}
			
			newFeatureVector.setValue(i, KernelSmoothing.kernelize(sim, width, kernelType));
		}
		
		return newFeatureVector;
	}
	
	private static SparseMatrix calculateUserSimilarity(int dataType, int simMethod, double smoothingFactor) {
		SparseMatrix result = new SparseMatrix(userCount+1, userCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			result.setValue(u, u, 1.0);
			for (int v = u+1; v <= userCount; v++) {
				SparseVector u_vec;
				SparseVector v_vec;
				double sim = 0.0;
				
				// Data Type:
				if (dataType == RATING_MATRIX) {
					u_vec = rateMatrix.getRowRef(u);
					v_vec = rateMatrix.getRowRef(v);
				}
				else if (dataType == MATRIX_FACTORIZATION) {
					u_vec = baseline.getU().getRowRef(u);
					v_vec = baseline.getU().getRowRef(v);
				}
				else { // Default: Rating Matrix
					u_vec = rateMatrix.getRowRef(u);
					v_vec = rateMatrix.getRowRef(v);
				}
				
				// Similarity Method:
				if (simMethod == PEARSON_CORR) { // Pearson correlation
					double u_avg = userRateAverage.getValue(u);
					double v_avg = userRateAverage.getValue(v);
					
					SparseVector a = u_vec.sub(u_avg);
					SparseVector b = v_vec.sub(v_avg);
					
					sim = a.innerProduct(b) / (a.norm() * b.norm());
				}
				else if (simMethod == VECTOR_COS) { // Vector cosine
					sim = u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm());
				}
				else if (simMethod == ARC_COS) { // Arccos
					sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
				}
				
				if (Double.isNaN(sim)) {
					sim = 0.0;
				}
				
				// Smoothing:
				if (smoothingFactor >= 0 && smoothingFactor <= 1) {
					sim = sim*(1 - smoothingFactor) + smoothingFactor;
				}
				
				result.setValue(u, v, sim);
				result.setValue(v, u, sim);
			}
		}
		
		return result;
	}
	
	private static SparseMatrix calculateItemSimilarity(int dataType, int simMethod, double smoothingFactor) {
		SparseMatrix result = new SparseMatrix(itemCount+1, itemCount+1);
		
		for (int i = 1; i <= itemCount; i++) {
			result.setValue(i, i, 1.0);
			for (int j = i+1; j <= itemCount; j++) {
				SparseVector i_vec;
				SparseVector j_vec;
				double sim = 0.0;
				
				// Data Type:
				if (dataType == RATING_MATRIX) {
					i_vec = rateMatrix.getRowRef(i);
					j_vec = rateMatrix.getRowRef(j);
				}
				else if (dataType == MATRIX_FACTORIZATION) {
					i_vec = baseline.getV().getColRef(i);
					j_vec = baseline.getV().getColRef(j);
				}
				else { // Default: Rating Matrix
					i_vec = rateMatrix.getRowRef(i);
					j_vec = rateMatrix.getRowRef(j);
				}
				
				// Similarity Method:
				if (simMethod == PEARSON_CORR) { // Pearson correlation
					double i_avg = userRateAverage.getValue(i);
					double j_avg = userRateAverage.getValue(j);
					
					SparseVector a = i_vec.sub(i_avg);
					SparseVector b = j_vec.sub(j_avg);
					
					sim = a.innerProduct(b) / (a.norm() * b.norm());
				}
				else if (simMethod == VECTOR_COS) { // Vector cosine
					sim = i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm());
				}
				else if (simMethod == ARC_COS) { // Arccos
					sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
					
					if (Double.isNaN(sim)) {
						sim = 0.0;
					}
				}
				
				if (Double.isNaN(sim)) {
					sim = 0.0;
				}
				
				// Smoothing:
				if (smoothingFactor >= 0 && smoothingFactor <= 1) {
					sim = sim*(1 - smoothingFactor) + smoothingFactor;
				}
				
				result.setValue(i, j, sim);
				result.setValue(j, i, sim);
			}
		}
	
		// print the result (matrix)
//		for (int i = 1; i <= 100; i++) {
//			String prt = "";
//			for (int j = 1; j <= 100; j++) {
//				prt += String.format("%.4f\t", result.getValue(i, j));
//			}
//			System.out.println(prt);
//		}
		// print the result (plain)
//		for (int i = 1; i <= itemCount; i++) {
//			for (int j = i+1; j <= itemCount; j++) {
//				System.out.println(String.format("%.4f\t", result.getValue(i, j)));
//			}
//		}
		
		return result;
	}
}