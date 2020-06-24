

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.FileOutputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.StringTokenizer;

import prea.data.splitter.*;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.data.structure.DenseMatrix;
import prea.data.structure.DenseVector;
import prea.recommender.*;
import prea.recommender.baseline.Average;
import prea.recommender.baseline.Constant;
import prea.recommender.baseline.ItemAverage;
import prea.recommender.baseline.Random;
import prea.recommender.baseline.UserAverage;
import prea.recommender.etc.FastNPCA;
import prea.recommender.etc.NonlinearPMF;
import prea.recommender.etc.RankBased;
import prea.recommender.etc.SlopeOne;
import prea.recommender.matrix.BayesianPMF;
import prea.recommender.matrix.NMF;
import prea.recommender.matrix.PMF;
import prea.recommender.matrix.RegularizedSVD;
import prea.recommender.memory.ItemBased;
import prea.recommender.memory.MemoryBasedRecommender;
import prea.recommender.memory.UserBased;
import prea.util.EvaluationMetrics;
import prea.util.SimpleEvaluationMetrics;
import prea.util.Printer;
import prea.util.Sort;

/**
 * A main class for ensemble experiments.
 * 
 * @author Joonseok Lee
 * @since 2012. 5. 16
 * @version 1.1
 */
public class Ensemble {
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
	/** Indicating whether to run all algorithms. */
	public static boolean runAllAlgorithms;
	/** The code for an algorithm which will run. */
	public static String algorithmCode;
	/** Parameter list for the algorithm to run. */
	public static String[] algorithmParameters;
	/** Indicating whether loading pre-calculated user similarity file or not */
	public static boolean userSimilarityPrefetch = false;
	/** Indicating whether loading pre-calculated item similarity file or not */
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
	
	private static double x_u_max;	// maximum number of ratings for a user
	private static double x_i_max;	// maximum number of ratings for an item
	private static double y_u_max;	// maximum standard deviation for a user
	private static double y_i_max;	// maximum standard deviation for an item
	private static SparseMatrix[] train;
	private static SparseMatrix[] test;
	private static int K;
	private static int L;
	private static double bestResult;
	private static boolean stopHere;
	private static SparseMatrix currEstimate;
	private static SparseMatrix currBeta;
	private static double[] itemRateVar;
	private static double maxItemRateVar;
	private static DenseMatrix userSimilarity;
	private static DenseMatrix itemSimilarity;
	private static double[] individualRMSE;
	private static SparseMatrix featuresTrain;
	private static SparseMatrix featuresTest;
	private static SparseVector featureMaxTrain;
	public static SparseVector userStdev;
	public static SparseVector itemStdev;
	
	/**
	 * Test examples for every algorithm. Also includes parsing the given parameters.
	 * 
	 * @param argv The argument list. Each element is separated by an empty space.
	 * First element is the data file name, and second one is the algorithm name.
	 * Third and later includes parameters for the chosen algorithm.
	 * Please refer to our web site for detailed syntax.
	 */
	public static void main(String argv[]) {
		// Set default setting first:
		dataFileName = "movieLens_1M";
		evaluationMode = DataSplitManager.SIMPLE_SPLIT;
		splitFileName = dataFileName + "_split.txt";
		testRatio = 0.2;
		foldCount = 5;
		runAllAlgorithms = true;
		
		// Parsing the argument:
		if (argv.length > 1) {
			parseCommandLine(argv);
		}
		
		// Read input file:
		readArff (dataFileName + ".arff");
		
		// Train/test data split:
		switch (evaluationMode) {
			case DataSplitManager.SIMPLE_SPLIT:
				SimpleSplit sSplit = new SimpleSplit(rateMatrix, testRatio, maxValue, minValue);
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
		}
	}
	
	/** Run an/all algorithm with given data, based on the setting from command arguments. */
	private static void run() {
		SparseVector indicator = new SparseVector(16);
		
		indicator.setValue(0, 1.0); // Median
		indicator.setValue(1, 1.0); // UserAvg
		indicator.setValue(2, 1.0); // ItemAvg
		indicator.setValue(3, 1.0); // UserBased
//		indicator.setValue(4, 1.0); // UserDefault
		indicator.setValue(5, 1.0); // ItemBased
//		indicator.setValue(6, 1.0); // ItemDefault
		indicator.setValue(7, 1.0); // Slope1
		indicator.setValue(8, 1.0); // RegSVD
		indicator.setValue(9, 1.0); // NMF
		indicator.setValue(10, 1.0); // PMF
		indicator.setValue(11, 1.0); // BPMF
		indicator.setValue(12, 1.0); // NLPMF
//		indicator.setValue(13, 1.0); // NPCA
//		indicator.setValue(14, 1.0); // RankMean
//		indicator.setValue(15, 1.0); // RankAsym
		
		individualRMSE = learnIndividual(indicator);
		//individualRMSE = readIndividual(indicator);
		
		// Method type, Level, Sample Count, max iter.
		int lv, smpCnt, maxIter, clusterCount;
		double flipRate, kernelWidth;
		
		lv = 1; smpCnt = 200; maxIter = 100; clusterCount = 2; flipRate = 0.05; kernelWidth = 0.9;
		System.out.println("6-99\tSmpCnt: " + smpCnt + "\tmaxIter: " + maxIter + "\tflipRate: " + flipRate + "\tkernelWidth: " + kernelWidth);
		stageWiseCombination3(6, lv, smpCnt, maxIter, clusterCount, flipRate, kernelWidth);
		
		lv = 1; smpCnt = 200; maxIter = 20; clusterCount = 2; flipRate = 0.05; kernelWidth = 0.05;
		System.out.println("0-DF\tSmpCnt: " + smpCnt + "\tmaxIter: " + maxIter);
		stageWiseCombination3(0, lv, smpCnt, maxIter, clusterCount, flipRate, kernelWidth);
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
	 * Test interface for a recommender system.
	 * Print MAE, RMSE, and rank-based half-life score for given test data.
	 * 
	 * @return evaluation metrics and elapsed time for learning and evaluation.
	 */
	public static String testRecommender(String algorithmName, Recommender r) {
		long learnStart = System.currentTimeMillis();
		r.buildModel(rateMatrix);
		long learnEnd = System.currentTimeMillis();
		
		long evalStart = System.currentTimeMillis();
		EvaluationMetrics result = r.evaluate(testMatrix);
		long evalEnd = System.currentTimeMillis();
		
		return algorithmName + "\t" + result.printOneLine() + "\t" + Printer.printTime(learnEnd - learnStart) + "\t" + Printer.printTime(evalEnd - evalStart);
	}
	
	/** Run one algorithm with customized parameters with given data. */
	public static void runIndividual(String algorithmCode, String[] parameters) {
		System.out.println(EvaluationMetrics.printTitle() + "\tTrain Time\tTest Time");
		
		if (algorithmCode.toLowerCase().equals("const")) {
			Constant constant = new Constant(userCount, itemCount, maxValue, minValue, 3.0);
			System.out.println(testRecommender("Const", constant));
		}
		else if (algorithmCode.toLowerCase().equals("avg")) {
			Average average = new Average(userCount, itemCount, maxValue, minValue);
			System.out.println(testRecommender("AllAvg", average));
		}
		else if (algorithmCode.toLowerCase().equals("useravg")) {
			UserAverage userAverage = new UserAverage(userCount, itemCount, maxValue, minValue, userRateAverage);
			System.out.println(testRecommender("UserAvg", userAverage));
		}
		else if (algorithmCode.toLowerCase().equals("itemavg")) {
			ItemAverage itemAverage = new ItemAverage(userCount, itemCount, maxValue, minValue, itemRateAverage);
			System.out.println(testRecommender("ItemAvg", itemAverage));
		}
		else if (algorithmCode.toLowerCase().equals("random")) {
			Random random = new Random(userCount, itemCount, maxValue, minValue);
			System.out.println(testRecommender("Random", random));
		}
		else if (algorithmCode.toLowerCase().equals("userbased")) {
			int neighborhoodSize;
			int similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
			boolean useDefaultValue = false;
			double defaultValue = 0.0;
			String userSimFileName = "";
			
			if (parameters.length < 1) {
				neighborhoodSize = 50;
			}
			else {
				neighborhoodSize = Integer.parseInt(parameters[0]);
				
				if (parameters.length < 2) {
					similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
				}
				else {
					if (parameters[1].equals("pearson")) similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
					else if (parameters[1].equals("cosine")) similarityMethod = MemoryBasedRecommender.VECTOR_COS;
					else if (parameters[1].equals("msd")) similarityMethod = MemoryBasedRecommender.MEAN_SQUARE_DIFF;
					else if (parameters[1].equals("mad")) similarityMethod = MemoryBasedRecommender.MEAN_ABS_DIFF;
					else if (parameters[1].equals("invuserfreq")) similarityMethod = MemoryBasedRecommender.INVERSE_USER_FREQUENCY;
					else similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
					
					if (parameters.length > 2) {
						for (int i = 2; i < parameters.length; i += 2) {
							if (parameters[i].equals("default")) {
								useDefaultValue = true;
								defaultValue = Double.parseDouble(parameters[i+1]);
							}
							else if (parameters[i].equals("usersim")) {
								userSimilarityPrefetch = true;
								userSimFileName = parameters[i+1];
							}
							else {
								// Do nothing. Pass the wrong command.
							}
						}
					}
				}
			}
			
			String algorithmName;
			if (useDefaultValue) algorithmName = "UserDft";
			else algorithmName = "UserBsd";
			
			UserBased userBsd = new UserBased(userCount, itemCount, maxValue, minValue, neighborhoodSize,
				similarityMethod, useDefaultValue, defaultValue, userRateAverage, userSimilarityPrefetch, userSimFileName);
			System.out.println(testRecommender(algorithmName, userBsd));
		}
		else if (algorithmCode.toLowerCase().equals("itembased")) {
			int neighborhoodSize;
			int similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
			boolean useDefaultValue = false;
			double defaultValue = 0.0;
			String itemSimFileName = "";
			
			if (parameters.length < 1) {
				neighborhoodSize = 50;
			}
			else {
				neighborhoodSize = Integer.parseInt(parameters[0]);
				
				if (parameters.length < 2) {
					similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
				}
				else {
					if (parameters[1].equals("pearson")) similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
					else if (parameters[1].equals("cosine")) similarityMethod = MemoryBasedRecommender.VECTOR_COS;
					else if (parameters[1].equals("msd")) similarityMethod = MemoryBasedRecommender.MEAN_SQUARE_DIFF;
					else if (parameters[1].equals("mad")) similarityMethod = MemoryBasedRecommender.MEAN_ABS_DIFF;
					else if (parameters[1].equals("invuserfreq")) similarityMethod = MemoryBasedRecommender.INVERSE_USER_FREQUENCY;
					else similarityMethod = MemoryBasedRecommender.PEARSON_CORR;
					
					if (parameters.length > 2) {
						for (int i = 2; i < parameters.length; i += 2) {
							if (parameters[i].equals("default")) {
								useDefaultValue = true;
								defaultValue = Double.parseDouble(parameters[i+1]);
							}
							else if (parameters[i].equals("usersim")) {
								itemSimilarityPrefetch = true;
								itemSimFileName = parameters[i+1];
							}
							else {
								// Do nothing. Pass the wrong command.
							}
						}
					}
				}
			}
			
			String algorithmName;
			if (useDefaultValue) algorithmName = "ItemDft";
			else algorithmName = "ItemBsd";
			
			ItemBased itemBsd = new ItemBased(userCount, itemCount, maxValue, minValue, neighborhoodSize,
				similarityMethod, useDefaultValue, defaultValue, itemRateAverage, itemSimilarityPrefetch, itemSimFileName);
			System.out.println(testRecommender(algorithmName, itemBsd));
		}
		else if (algorithmCode.toLowerCase().equals("slopeone")) {
			SlopeOne slope1 = new SlopeOne(userCount, itemCount, maxValue, minValue);
			System.out.println(testRecommender("Slope1", slope1));
		}
		else if (algorithmCode.toLowerCase().equals("regsvd")) {
			RegularizedSVD regsvd = new RegularizedSVD(userCount, itemCount, maxValue, minValue, Integer.parseInt(parameters[0]), Double.parseDouble(parameters[1]), Double.parseDouble(parameters[2]), 0, Integer.parseInt(parameters[3]), false);
			System.out.println(testRecommender("RegSVD", regsvd));
		}
		else if (algorithmCode.toLowerCase().equals("nmf")) {
			// Direct evaluation
			NMF nmf = new NMF(userCount, itemCount, maxValue, minValue, Integer.parseInt(parameters[0]), 0, Double.parseDouble(parameters[1]), 0, Integer.parseInt(parameters[2]), 0.2, false);
			System.out.println(testRecommender("NMF", nmf));
			
			// Saving in Serialized file
			try {
				FileOutputStream f = new FileOutputStream("nmf.obj"); 
				ObjectOutput s = new ObjectOutputStream(f); 
				s.writeObject(nmf);
				s.flush();
				f.close();
			}
			catch(IOException e) {
				System.out.println(e);
			}
			
			//===========================================================
			// Reading from Serialized file
//			try {
//				FileInputStream f = new FileInputStream("nmf.obj");
//				ObjectInput s = new ObjectInputStream(f);
//				NMF nmf = (NMF) s.readObject();
//				
//				f.close();
//				
//				nmf.evaluate(rateMatrix);
//				
//				long evalStart = System.currentTimeMillis();
//				EvaluationMetrics testResult = nmf.evaluate(testMatrix);
//				long evalEnd = System.currentTimeMillis();
//				
//				System.out.println("NMF\t" + testResult.printOneLine() + "\tN/A\t" + Printer.printTime(evalEnd - evalStart));
//			}
//			catch(Exception e) {}
		}
		else if (algorithmCode.toLowerCase().equals("pmf")) {
			PMF pmf = new PMF(userCount, itemCount, maxValue, minValue, Integer.parseInt(parameters[0]), Integer.parseInt(parameters[1]), Double.parseDouble(parameters[2]), Double.parseDouble(parameters[3]), Integer.parseInt(parameters[4]), false);
			System.out.println(testRecommender("PMF", pmf));
		}
		else if (algorithmCode.toLowerCase().equals("bpmf")) {
			BayesianPMF bpmf = new BayesianPMF(userCount, itemCount, maxValue, minValue, Integer.parseInt(parameters[0]), 0, 0, 0, Integer.parseInt(parameters[1]), false);
			System.out.println(testRecommender("BPMF", bpmf));
		}
		else if (algorithmCode.toLowerCase().equals("nlpmf")) {
			NonlinearPMF nlpmf = new NonlinearPMF(userCount, itemCount, maxValue, minValue, Integer.parseInt(parameters[0]), Double.parseDouble(parameters[1]), Double.parseDouble(parameters[2]), Integer.parseInt(parameters[3]),
					Double.parseDouble(parameters[4]),Double.parseDouble(parameters[5]),Double.parseDouble(parameters[6]),Double.parseDouble(parameters[7]));
			System.out.println(testRecommender("NLPMF", nlpmf));
		}
		else if (algorithmCode.toLowerCase().equals("npca")) {
			FastNPCA npca = new FastNPCA(userCount, itemCount, maxValue, minValue, Double.parseDouble(parameters[0]), Integer.parseInt(parameters[1]));
			System.out.println(testRecommender("NPCA", npca));
		}
		else if (algorithmCode.toLowerCase().equals("rank")) {
			RankBased rnkMean = new RankBased(userCount, itemCount, maxValue, minValue, Double.parseDouble(parameters[0]), RankBased.MEAN_LOSS);
			System.out.println(testRecommender("RnkMean", rnkMean));
			
			RankBased rnkAsym = new RankBased(userCount, itemCount, maxValue, minValue, Double.parseDouble(parameters[0]), RankBased.ASYMM_LOSS);
			System.out.println(testRecommender("RnkAsym", rnkAsym));
		}
	}
	
	/** Run all algorithms with given data. */
	public static void runAll() {
		System.out.println(EvaluationMetrics.printTitle() + "\tTrain Time\tTest Time");
		
		Constant constant = new Constant(userCount, itemCount, maxValue, minValue, 3.0);
		System.out.println(testRecommender("Const", constant));
		
		Average average = new Average(userCount, itemCount, maxValue, minValue);
		System.out.println(testRecommender("AllAvg", average));
		
		UserAverage userAverage = new UserAverage(userCount, itemCount, maxValue, minValue, userRateAverage);
		System.out.println(testRecommender("UserAvg", userAverage));
		
		ItemAverage itemAverage = new ItemAverage(userCount, itemCount, maxValue, minValue, itemRateAverage);
		System.out.println(testRecommender("ItemAvg", itemAverage));
		
		Random random = new Random(userCount, itemCount, maxValue, minValue);
		System.out.println(testRecommender("Random", random));
		
		UserBased userBsd = new UserBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, false, 0.0, userRateAverage, userSimilarityPrefetch, dataFileName + "_userSim.txt");
		System.out.println(testRecommender("UserBsd", userBsd));
		
		UserBased userDft = new UserBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, true, (maxValue + minValue) / 2, userRateAverage, userSimilarityPrefetch, dataFileName + "_userSim.txt");
		System.out.println(testRecommender("UserDft", userDft));
		
		ItemBased itemBsd = new ItemBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, false, 0.0, itemRateAverage, itemSimilarityPrefetch, dataFileName + "_itemSim.txt");
		System.out.println(testRecommender("ItemBsd", itemBsd));
		
		ItemBased itemDft = new ItemBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, true, (maxValue + minValue) / 2, itemRateAverage, itemSimilarityPrefetch, dataFileName + "_itemSim.txt");
		System.out.println(testRecommender("ItemDft", itemDft));
		
		SlopeOne slope1 = new SlopeOne(userCount, itemCount, maxValue, minValue);
		System.out.println(testRecommender("Slope1", slope1));
		
		RegularizedSVD regsvd = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 60, 0.005, 0.1, 0, 200, false);
		System.out.println(testRecommender("RegSVD", regsvd));
		
		NMF nmf = new NMF(userCount, itemCount, maxValue, minValue, 100, 0, 0.0001, 0, 50, 0.2, false);
		System.out.println(testRecommender("NMF", nmf));
		
		PMF pmf = new PMF(userCount, itemCount, maxValue, minValue, 10, 50, 0.4, 0.8, 200, false);
		System.out.println(testRecommender("PMF", pmf));
		
		BayesianPMF bpmf = new BayesianPMF(userCount, itemCount, maxValue, minValue, 2, 0, 0, 0, 20, false);
		System.out.println(testRecommender("BPMF", bpmf));
		
		NonlinearPMF nlpmf = new NonlinearPMF(userCount, itemCount, maxValue, minValue, 10, 0.0001, 0.9, 2, 1, 1, 0.11, 5);
		System.out.println(testRecommender("NLPMF", nlpmf));
		
		FastNPCA npca = new FastNPCA(userCount, itemCount, maxValue, minValue, 0.15, 50);
		System.out.println(testRecommender("NPCA", npca));
		
		RankBased rnkMean = new RankBased(userCount, itemCount, maxValue, minValue, 0.02, RankBased.MEAN_LOSS);
		System.out.println(testRecommender("RnkMean", rnkMean));
		
		RankBased rnkAsym = new RankBased(userCount, itemCount, maxValue, minValue, 0.02, RankBased.ASYMM_LOSS);
		System.out.println(testRecommender("RnkAsym", rnkAsym));
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
			rateMatrix = new SparseMatrix(500000, itemCount+1); // max 480189, 17770
			
			// Read data:
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					line = line.substring(1, line.length() - 1);
					
					StringTokenizer st = new StringTokenizer (line, ",");
					
					double rateSum = 0.0;
					int rateCount = 0;
					int userID = 0;
					
					while (st.hasMoreTokens()) {
						String token = st.nextToken().trim();
						int i = token.indexOf(" ");
						
						int movieID, rate;
						int index = Integer.parseInt(token.substring(0, i));
						String data = token.substring(i+1);
						
						if (index == 0) { // User ID
							userID = Integer.parseInt(data);
							
							rateSum = 0.0;
							rateCount = 0;
							
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
							
							rateSum += rate;
							rateCount++;
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
	
	public static double[] readIndividual(SparseVector indicator) {
		setMax();
		
		int index = 0;
		K = indicator.itemCount();
		train = new SparseMatrix[K];
		test = new SparseMatrix[K];
		System.out.println(EvaluationMetrics.printTitle());
		
		double[] rmse = new double[K];
		
		for (int i = 0; i < 16; i++) {
			if (indicator.getValue(i) != 0) {
				train[index] = readPrecalculatedFile("train" + i + ".txt");
				test[index] = readPrecalculatedFile("test" + i + ".txt");
				SimpleEvaluationMetrics testEval = new SimpleEvaluationMetrics(testMatrix, test[index], 5, 1);
				System.out.println("Alg" + i + "\t" + testEval.printOneLine());
				rmse[index] = testEval.getRMSE();
				index++;
			}
		}
		
		// Calculate variance of ratings over algorithms:
		double[][] relativePerfUser = new double[index][userCount+1];
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector avg = new SparseVector(itemCount+1);
			SparseVector var = new SparseVector(itemCount+1);
			
			for (int a = 0; a < index; a++) {
				avg = avg.plus(train[a].getRowRef(u));
			}
			
			if (avg.sum() == 0) {
				for (int a = 0; a < index; a++) {
					relativePerfUser[a][u] = 0.0;
				}
			}
			else {
				avg = avg.scale(1.0/(int) index);
				
				for (int a = 0; a < index; a++) {
					var = avg.minus(train[a].getRowRef(u));
					var = var.power(2.0);
					relativePerfUser[a][u] = Math.sqrt(var.average());
				}
			}
		}
		
		itemRateVar = new double[itemCount+1];
		maxItemRateVar = 0.0;
		double[][] relativePerfItem = new double[index][itemCount+1];
		
		for (int i = 1; i <= itemCount; i++) {
			SparseVector avg = new SparseVector(userCount+1);
			SparseVector var = new SparseVector(userCount+1);
			
			for (int a = 0; a < index; a++) {
				avg = avg.plus(train[a].getColRef(i));
			}
			
			if (avg.sum() == 0) {
				itemRateVar[i] = 0.0;
				
				for (int a = 0; a < index; a++) {
					relativePerfItem[a][i] = 0.0;
				}
			}
			else {
				avg = avg.scale(1.0/(int) index);
				
				for (int a = 0; a < index; a++) {
					var = avg.minus(train[a].getColRef(i));
					var = var.power(2.0);
					relativePerfItem[a][i] = Math.sqrt(var.average());
				}
				
				if (var.sum() == 0.0) {
					itemRateVar[i] = 0.0;
				}
				else {
					itemRateVar[i] = var.average();
				}
			}
			
			if (itemRateVar[i] > maxItemRateVar) {
				maxItemRateVar = itemRateVar[i];
			}
		}
		
//		// Print relative strength of each algorithm on each user/item:
//		for (int a = 0; a < index; a++) {
//			String prt = a + "U";
//			for (int u = 1; u <= userCount; u++) {
//				prt += "\t" + String.format("%.4f", relativePerfUser[a][u]);
//			}
//			System.out.println(prt);
//		}
//		
//		for (int a = 0; a < index; a++) {
//			String prt = a + "I";
//			for (int i = 1; i <= itemCount; i++) {
//				prt += "\t" + String.format("%.4f", relativePerfItem[a][i]);
//			}
//			System.out.println(prt);
//		}
		
		return rmse;
	}
	
	public static double[] learnIndividual(SparseVector indicator) {
		setMax();
		
		int index = 0;
		K = indicator.itemCount();
		train = new SparseMatrix[K];
		test = new SparseMatrix[K];
		System.out.println(EvaluationMetrics.printTitle());
		
		double[] rmse = new double[K];
		
		// Build individual models:
		if (indicator.getValue(0) != 0) {
			Constant constant = new Constant(userCount, itemCount, maxValue, minValue, 3.0);
			constant.buildModel(rateMatrix);
			EvaluationMetrics constantTrain = constant.evaluate(rateMatrix);
			EvaluationMetrics constantTest = constant.evaluate(testMatrix);
			train[index] = constantTrain.getPrediction();
			test[index] = constantTest.getPrediction();
			System.out.println("Const" + "\t" + constantTest.printOneLine());
			rmse[index] = constantTest.getRMSE();
			index++;
		}
		if (indicator.getValue(1) != 0) {
			UserAverage userAverage = new UserAverage(userCount, itemCount, maxValue, minValue, userRateAverage);
			userAverage.buildModel(rateMatrix);
			EvaluationMetrics userAvgTrain = userAverage.evaluate(rateMatrix);
			EvaluationMetrics userAvgTest = userAverage.evaluate(testMatrix);
			train[index] = userAvgTrain.getPrediction();
			test[index] = userAvgTest.getPrediction();
			System.out.println("UserAvg" + "\t" + userAvgTest.printOneLine());
			rmse[index] = userAvgTest.getRMSE();
			index++;
		}
		if (indicator.getValue(2) != 0) {
			ItemAverage itemAverage = new ItemAverage(userCount, itemCount, maxValue, minValue, itemRateAverage);
			itemAverage.buildModel(rateMatrix);
			EvaluationMetrics itemAvgTrain = itemAverage.evaluate(rateMatrix);
			EvaluationMetrics itemAvgTest = itemAverage.evaluate(testMatrix);
			train[index] = itemAvgTrain.getPrediction();
			test[index] = itemAvgTest.getPrediction();
			System.out.println("ItemAvg" + "\t" + itemAvgTest.printOneLine());
			rmse[index] = itemAvgTest.getRMSE();
			index++;
		}
		if (indicator.getValue(3) != 0) {
			UserBased userBsd = new UserBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, false, 0.0, userRateAverage, userSimilarityPrefetch, dataFileName + "_userSim.txt");
			userBsd.buildModel(rateMatrix);
			EvaluationMetrics userBasedTrain = userBsd.evaluate(rateMatrix);
			EvaluationMetrics userBasedTest = userBsd.evaluate(testMatrix);
			train[index] = userBasedTrain.getPrediction();
			test[index] = userBasedTest.getPrediction();
			System.out.println("UserBsd" + "\t" + userBasedTest.printOneLine());
			rmse[index] = userBasedTest.getRMSE();
			index++;
		}
		if (indicator.getValue(4) != 0) {
			UserBased userDft = new UserBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, true, (maxValue + minValue) / 2, userRateAverage, userSimilarityPrefetch, dataFileName + "_userSim.txt");
			userDft.buildModel(rateMatrix);
			EvaluationMetrics userDefaultTrain = userDft.evaluate(rateMatrix);
			EvaluationMetrics userDefaultTest = userDft.evaluate(testMatrix);
			train[index] = userDefaultTrain.getPrediction();
			test[index] = userDefaultTest.getPrediction();
			System.out.println("UserDft" + "\t" + userDefaultTest.printOneLine());
			rmse[index] = userDefaultTest.getRMSE();
			index++;
		}
		if (indicator.getValue(5) != 0) {
			ItemBased itemBsd = new ItemBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, false, 0.0, itemRateAverage, itemSimilarityPrefetch, dataFileName + "_itemSim.txt");
			itemBsd.buildModel(rateMatrix);
			EvaluationMetrics itemBasedTrain = itemBsd.evaluate(rateMatrix);
			EvaluationMetrics itemBasedTest = itemBsd.evaluate(testMatrix);
			train[index] = itemBasedTrain.getPrediction();
			test[index] = itemBasedTest.getPrediction();
			System.out.println("ItemBsd" + "\t" + itemBasedTest.printOneLine());
			rmse[index] = itemBasedTest.getRMSE();
			index++;
		}
		if (indicator.getValue(6) != 0) {
			ItemBased itemDft = new ItemBased(userCount, itemCount, maxValue, minValue, 50, MemoryBasedRecommender.PEARSON_CORR, true, (maxValue + minValue) / 2, itemRateAverage, itemSimilarityPrefetch, dataFileName + "_itemSim.txt");
			itemDft.buildModel(rateMatrix);
			EvaluationMetrics itemDefaultTrain = itemDft.evaluate(rateMatrix);
			EvaluationMetrics itemDefaultTest = itemDft.evaluate(testMatrix);
			train[index] = itemDefaultTrain.getPrediction();
			test[index] = itemDefaultTest.getPrediction();
			System.out.println("ItemDft" + "\t" + itemDefaultTest.printOneLine());
			rmse[index] = itemDefaultTest.getRMSE();
			index++;
		}
		if (indicator.getValue(7) != 0) {
			SlopeOne slope1 = new SlopeOne(userCount, itemCount, maxValue, minValue);
			slope1.buildModel(rateMatrix);
			EvaluationMetrics slopeOneTrain = slope1.evaluate(rateMatrix);
			EvaluationMetrics slopeOneTest = slope1.evaluate(testMatrix);
			train[index] = slopeOneTrain.getPrediction();
			test[index] = slopeOneTest.getPrediction();
			System.out.println("Slope1" + "\t" + slopeOneTest.printOneLine());
			rmse[index] = slopeOneTest.getRMSE();
			index++;
		}
		if (indicator.getValue(8) != 0) {
			RegularizedSVD regsvd = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 60, 0.005, 0.1, 0, 200, false);
			regsvd.buildModel(rateMatrix);
			EvaluationMetrics regSVDTrain = regsvd.evaluate(rateMatrix);
			EvaluationMetrics regSVDTest = regsvd.evaluate(testMatrix);
			train[index] = regSVDTrain.getPrediction();
			test[index] = regSVDTest.getPrediction();
			System.out.println("RegSVD" + "\t" + regSVDTest.printOneLine());
			rmse[index] = regSVDTest.getRMSE();
			index++;
		}
		if (indicator.getValue(9) != 0) {
			NMF nmf = new NMF(userCount, itemCount, maxValue, minValue, 100, 0, 0.0001, 0, 50, 0.2, false);
			nmf.buildModel(rateMatrix);
			EvaluationMetrics nmfTrain = nmf.evaluate(rateMatrix);
			EvaluationMetrics nmfTest = nmf.evaluate(testMatrix);
			train[index] = nmfTrain.getPrediction();
			test[index] = nmfTest.getPrediction();
			System.out.println("NMF" + "\t" + nmfTest.printOneLine());
			rmse[index] = nmfTest.getRMSE();
			index++;
		}
		if (indicator.getValue(10) != 0) {
			PMF pmf = new PMF(userCount, itemCount, maxValue, minValue, 10, 50, 0.4, 0.8, 200, false);
			pmf.buildModel(rateMatrix);
			EvaluationMetrics pmfTrain = pmf.evaluate(rateMatrix);
			EvaluationMetrics pmfTest = pmf.evaluate(testMatrix);
			train[index] = pmfTrain.getPrediction();
			test[index] = pmfTest.getPrediction();
			System.out.println("PMF" + "\t" + pmfTest.printOneLine());
			rmse[index] = pmfTest.getRMSE();
			index++;
		}
		if (indicator.getValue(11) != 0) {
			BayesianPMF bpmf = new BayesianPMF(userCount, itemCount, maxValue, minValue, 2, 0, 0, 0, 20, false);
			bpmf.buildModel(rateMatrix);
			EvaluationMetrics bpmfTrain = bpmf.evaluate(rateMatrix);
			EvaluationMetrics bpmfTest = bpmf.evaluate(testMatrix);
			train[index] = bpmfTrain.getPrediction();
			test[index] = bpmfTest.getPrediction();
			System.out.println("BPMF" + "\t" + bpmfTest.printOneLine());
			rmse[index] = bpmfTest.getRMSE();
			index++;
		}
		if (indicator.getValue(12) != 0) {
			NonlinearPMF nlpmf = new NonlinearPMF(userCount, itemCount, maxValue, minValue, 10, 0.0001, 0.9, 2, 1, 1, 0.11, 5);
			nlpmf.buildModel(rateMatrix);
			EvaluationMetrics nlpmfTrain = nlpmf.evaluate(rateMatrix);
			EvaluationMetrics nlpmfTest = nlpmf.evaluate(testMatrix);
			train[index] = nlpmfTrain.getPrediction();
			test[index] = nlpmfTest.getPrediction();
			System.out.println("NLPMF" + "\t" + nlpmfTest.printOneLine());
			rmse[index] = nlpmfTest.getRMSE();
			index++;
		}
		if (indicator.getValue(13) != 0) {
			FastNPCA npca = new FastNPCA(userCount, itemCount, maxValue, minValue, 0.15, 50);
			npca.buildModel(rateMatrix);
			EvaluationMetrics npcaTrain = npca.evaluate(rateMatrix);
			EvaluationMetrics npcaTest = npca.evaluate(testMatrix);
			train[index] = npcaTrain.getPrediction();
			test[index] = npcaTest.getPrediction();
			System.out.println("NPCA" + "\t" + npcaTest.printOneLine());
			rmse[index] = npcaTest.getRMSE();
			index++;
		}
		if (indicator.getValue(14) != 0) {
			RankBased rnkMean = new RankBased(userCount, itemCount, maxValue, minValue, 0.02, RankBased.MEAN_LOSS);
			EvaluationMetrics rankBasedMeanTrain = rnkMean.evaluate(rateMatrix);
			EvaluationMetrics rankBasedMeanTest = rnkMean.evaluate(testMatrix);
			train[index] = rankBasedMeanTrain.getPrediction();
			test[index] = rankBasedMeanTest.getPrediction();
			System.out.println("RnkMean" + "\t" + rankBasedMeanTest.printOneLine());
			rmse[index] = rankBasedMeanTest.getRMSE();
			index++;
		}
		if (indicator.getValue(15) != 0) {
			RankBased rnkAsym = new RankBased(userCount, itemCount, maxValue, minValue, 0.02, RankBased.ASYMM_LOSS);
			EvaluationMetrics rankBasedAsymTrain = rnkAsym.evaluate(rateMatrix);
			EvaluationMetrics rankBasedAsymTest = rnkAsym.evaluate(testMatrix);
			train[index] = rankBasedAsymTrain.getPrediction();
			test[index] = rankBasedAsymTest.getPrediction();
			System.out.println("RnkAsym" + "\t" + rankBasedAsymTest.printOneLine());
			rmse[index] = rankBasedAsymTest.getRMSE();
			index++;
		}
		
		// Calculate variance of ratings over algorithms:
		double[][] relativePerfUser = new double[index][userCount+1];
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector avg = new SparseVector(itemCount+1);
			SparseVector var = new SparseVector(itemCount+1);
			
			for (int a = 0; a < index; a++) {
				avg = avg.plus(train[a].getRowRef(u));
			}
			
			if (avg.sum() == 0) {
				for (int a = 0; a < index; a++) {
					relativePerfUser[a][u] = 0.0;
				}
			}
			else {
				avg = avg.scale(1.0/(int) index);
				
				for (int a = 0; a < index; a++) {
					var = avg.minus(train[a].getRowRef(u));
					var = var.power(2.0);
					relativePerfUser[a][u] = Math.sqrt(var.average());
				}
			}
		}
		
		itemRateVar = new double[itemCount+1];
		maxItemRateVar = 0.0;
		double[][] relativePerfItem = new double[index][itemCount+1];
		
		for (int i = 1; i <= itemCount; i++) {
			SparseVector avg = new SparseVector(userCount+1);
			SparseVector var = new SparseVector(userCount+1);
			
			for (int a = 0; a < index; a++) {
				avg = avg.plus(train[a].getColRef(i));
			}
			
			if (avg.sum() == 0) {
				itemRateVar[i] = 0.0;
				
				for (int a = 0; a < index; a++) {
					relativePerfItem[a][i] = 0.0;
				}
			}
			else {
				avg = avg.scale(1.0/(int) index);
				
				for (int a = 0; a < index; a++) {
					var = avg.minus(train[a].getColRef(i));
					var = var.power(2.0);
					relativePerfItem[a][i] = Math.sqrt(var.average());
				}
				
				if (var.sum() == 0.0) {
					itemRateVar[i] = 0.0;
				}
				else {
					itemRateVar[i] = var.average();
				}
			}
			
			if (itemRateVar[i] > maxItemRateVar) {
				maxItemRateVar = itemRateVar[i];
			}
		}
		
//		// Print relative strength of each algorithm on each user/item:
//		for (int a = 0; a < index; a++) {
//			String prt = a + "U";
//			for (int u = 1; u <= userCount; u++) {
//				prt += "\t" + String.format("%.4f", relativePerfUser[a][u]);
//			}
//			System.out.println(prt);
//		}
//		
//		for (int a = 0; a < index; a++) {
//			String prt = a + "I";
//			for (int i = 1; i <= itemCount; i++) {
//				prt += "\t" + String.format("%.4f", relativePerfItem[a][i]);
//			}
//			System.out.println(prt);
//		}
		
		return rmse;
	}
	
	public static void setMax() {
		x_u_max = 0;
		y_u_max = 0;
		for (int u = 1; u <= userCount; u++) {
			if (rateMatrix.getRowRef(u).itemCount() > x_u_max) {
				x_u_max = rateMatrix.getRowRef(u).itemCount();
			}
			if (rateMatrix.getRowRef(u).stdev() > y_u_max) {
				y_u_max = rateMatrix.getRowRef(u).stdev();
			}
		}
		
		x_i_max = 0;
		y_i_max = 0;
		for (int i = 1; i <= itemCount; i++) {
			if (rateMatrix.getColRef(i).itemCount() > x_i_max) {
				x_i_max = rateMatrix.getColRef(i).itemCount();
			}
			if (rateMatrix.getColRef(i).stdev() > y_i_max) {
				y_i_max = rateMatrix.getColRef(i).stdev();
			}
		}
	}
	
	// Indicator-feature Search version:
	public static void stageWiseCombination3(int method, int lv, int smpCnt, int maxIter, int clusterCount, double flipRate, double kernelWidth) {
		// Calculating user/item similarity:
		calculateSimilarity(2);
		
		// NV: Indicator-variables selecting good items from train set.
		int[] selectedAlgo = new int[maxIter];
		SparseVector[] selectedFeat = new SparseVector[maxIter];
		boolean[] selectedFeatIsItem = new boolean[maxIter];
		selectedFeat[0] = new SparseVector(itemCount+1);
		
		currBeta = new SparseMatrix(maxIter, 2);
		currEstimate = new SparseMatrix(userCount+1, itemCount+1);
		bestResult = Double.MAX_VALUE;
		stopHere = false;
		
		int bestAlgorithm = -1;
		SparseVector bestFeature;
		boolean bestFeatureIsItemFeature;
		SimpleEvaluationMetrics savedMetrics = new SimpleEvaluationMetrics(testMatrix, testMatrix, 5, 1);
		SimpleEvaluationMetrics[] bestMetrics = new SimpleEvaluationMetrics[K];
		double bestRMSE = Double.MAX_VALUE;
		DenseVector bestBeta = new DenseVector(2);
		
		int stopCount = 0;
		double prevRMSE = Double.MAX_VALUE;

		// N2: Indicator-variables
		if (method >= 1 && method <= 6) {
			selectedAlgo = new int[maxIter];
			selectedFeat = new SparseVector[maxIter];
			selectedFeat[0] = new SparseVector(itemCount+1);
			
			currBeta = new SparseMatrix(maxIter, 2);
			currEstimate = new SparseMatrix(userCount+1, itemCount+1);
			bestResult = Double.MAX_VALUE;
			stopHere = false;
			
			bestAlgorithm = -1;
			bestFeature = null;
			bestFeatureIsItemFeature = true;
			savedMetrics = new SimpleEvaluationMetrics(testMatrix, testMatrix, 5, 1);
			bestRMSE = Double.MAX_VALUE;
			bestBeta = new DenseVector(2);
			
			stopCount = 0;
			
			int totalIter = 0;
			for (int t = 0; t < maxIter; t++) {
				if (t >= maxIter || stopHere) {
					break;
				}
				else if (t == 0) {
					// Select algorithm 1 always:
//					selectedAlgo[t] = 1;
					
					// Select the first algorithm, with smallest RMSE:
					double bestIndividual = Double.MAX_VALUE;
					for (int k = 0; k < K; k++) {
						if (individualRMSE[k] < bestIndividual) {
							bestIndividual = individualRMSE[k];
							selectedAlgo[t] = k;
						}
					}
					
					// Select the first algorithm, with largest RMSE:
//					double bestIndividual = Double.MIN_VALUE;
//					for (int k = 0; k < K; k++) {
//						if (individualRMSE[k] > bestIndividual) {
//							bestIndividual = individualRMSE[k];
//							selectedAlgo[t] = k;
//						}
//					}
					
					currBeta.setValue(t, 0, 1.0);
					currBeta.setValue(t, 1, 0.0);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								currEstimate.setValue(u, i, test[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
					
					EvaluationMetrics indv = new EvaluationMetrics(testMatrix, currEstimate, 5, 1);
					System.out.println("MixN2\t" + indv.printOneLine() + "\t" + selectedAlgo[0]/* + "\tx"*/);
					
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								currEstimate.setValue(u, i, train[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
				}
				else {
					bestAlgorithm = -1;
					bestFeature = null;
					bestRMSE = Double.MAX_VALUE;
					bestBeta = new DenseVector(2);
					
					// Initialize sample list of features:
					int sampleCount = smpCnt;
					SparseVector[] oldFeatureList = new SparseVector[sampleCount];
					SparseVector[] newFeatureList = new SparseVector[sampleCount];
					for (int l = 0; l < sampleCount; l++) {
						if (method == 4) {
							oldFeatureList[l] = getVarianceBasedFeatureVector(itemCount+1, flipRate - t * 0.002);
						}
						else {
							oldFeatureList[l] = getRandomFeatureVector(itemCount+1, flipRate);
						}
					}
					
					double[] currPerformance = new double[sampleCount];
					double[] prevPerformance = new double[sampleCount];
					int[] prevPerfIndex = new int[sampleCount];
					
					for (int level = 1; level <= lv; level++) {
						boolean[] isItemFeature = new boolean[sampleCount];
						for (int l = 0; l < sampleCount; l++) {
							SparseVector feature;
							isItemFeature[l] = true;
							
							// Method 1: using randomly chosen 50 features for each level
							if (method == 1) {
								feature = getRandomFeatureVector(itemCount+1, flipRate);
							}
							// Method 5: Kernel Smoothing
							else if (method == 5) {
								feature = getKernelFeatureVector(itemCount+1, flipRate, kernelWidth, true);
							}
							// Method 6: Kernel Smoothing with both user/item features:
							else { // method == 6)
								if (Math.random() > 0.5) {
									isItemFeature[l] = false;
									feature = getKernelFeatureVector(userCount+1, flipRate, kernelWidth, isItemFeature[l]);
								}
								else {
									isItemFeature[l] = true;
									feature = getKernelFeatureVector(itemCount+1, flipRate, kernelWidth, isItemFeature[l]);
								}
								feature.scale(1.0 / (double) feature.itemCount());
							}
							
							newFeatureList[l] = feature;
							currPerformance[l] = Double.MAX_VALUE;
							
							for (int k = 0; k < K; k++) { // for all candidate algorithms
								SparseMatrix diffK = train[k].plus(currEstimate.scale(-1));
	
								//double[] lambda0 = {1E-4, 1E0, 1E4, 1E8, 1E12};
								double[] lambda0 = {1E-4, 1E4};
								for (int l0 = 0; l0 < lambda0.length; l0++) {
									// build intermediate data:
									SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
	
									int M = 2;							// feature count
									int N = rateMatrix.itemCount();		// train point count
	
									DenseMatrix Z = new DenseMatrix(N, M);
									DenseVector r = new DenseVector(N);
	
									int n = 0;
									for (int u = 1; u <= userCount; u++) {
										SparseVector userRating = rateMatrix.getRowRef(u);
										int[] indexList = userRating.indexList();
										
										if (indexList != null) {
											for (int i : indexList) {
												double diff = diffK.getValue(u, i);
												Z.setValue(n, 0, diff);
												if (isItemFeature[l]) {
													Z.setValue(n, 1, diff * feature.getValue(i));
												}
												else {
													Z.setValue(n, 1, diff * feature.getValue(u));
												}
	
												r.setValue(n, rateMatrix.getValue(u, i) - currEstimate.getValue(u, i));
												n++;
											}
										}
									}
	
									DenseVector b = learnBeta(Z, r, lambda0[l0], lambda0[l0], 2);
	
									// check for constraints and update:
									n = 0;
									boolean done = true;
									int[] changed = new int[2];
									for (int u = 1; u <= userCount; u++) {
										SparseVector userRating = rateMatrix.getRowRef(u);
										int[] indexList = userRating.indexList();
	
										if (indexList != null) {
											for (int i : indexList) {
												double diff = diffK.getValue(u, i);
	
												if (b.getValue(0) + b.getValue(1) < 0) {
													double updateValue;
													if (isItemFeature[l]) {
														updateValue = feature.getValue(i) * diff;
													}
													else {
														updateValue = feature.getValue(u) * diff;
													}
													Z.setValue(n, 0, Z.getValue(n, 0) - updateValue);
													Z.setValue(n, 1, 0.0);
													changed[1] = -1;
													done = false;
												}
												else if (b.getValue(0) + b.getValue(1) > 1) {
													double updateValue;
													if (isItemFeature[l]) {
														updateValue = feature.getValue(i) * diff;
													}
													else {
														updateValue = feature.getValue(u) * diff;
													}
													Z.setValue(n, 0, Z.getValue(n, 0) - updateValue);
													Z.setValue(n, 1, 0.0);
													r.setValue(n, r.getValue(n) - updateValue);
													changed[1] = 1;
													done = false;
												}
	
												if (b.getValue(0) < 0) {
													if (changed[1] != 0) {
														if (changed[1] > 0) b.setValue(1, 1.0);
														else b.setValue(1, 0.0);
	
														b.setValue(0, 0.0);
														done = true;
													}
													else {
														Z.setValue(n, 0, 0.0);
													}
												}
												else if (b.getValue(0) > 1) {
													if (changed[1] != 0) {
														if (changed[1] > 0) b.setValue(1, 0.0);
														else b.setValue(1, -1.0);
	
														b.setValue(0, 1.0);
														done = true;
													}
													else {
														r.setValue(n, r.getValue(n) - Z.getValue(n, 0));
														Z.setValue(n, 0, 0.0);
													}
												}
	
												n++;
											}
										}
									}
									
									if (!done) {
										b = learnBeta(Z, r, lambda0[l0], lambda0[l0], 2);
									}
									
									// estimation (test error):
									n = 0;
									predicted = new SparseMatrix(userCount+1, itemCount+1);
									for (int u = 1; u <= userCount; u++) {
										SparseVector userRating = testMatrix.getRowRef(u);
										int[] indexList = userRating.indexList();
										
										if (indexList != null) {
											for (int i : indexList) {
												// Use sampled test items only:
												double rdm = Math.random();
												if (rdm < 1.1) {
													double alpha;
													if (isItemFeature[l]) {
														alpha = b.getValue(0) + b.getValue(1) * feature.getValue(i);
													}
													else {
														alpha = b.getValue(0) + b.getValue(1) * feature.getValue(u);
													}
													
													double estimate = alpha * test[k].getValue(u, i)
																+ (1 - alpha) * currEstimate.getValue(u, i);
													
													predicted.setValue(u, i, estimate);
												}
												n++;
											}
										}
									}
									
									// Compare result:
									SimpleEvaluationMetrics mixTest = new SimpleEvaluationMetrics(testMatrix, predicted, 5, 1);
									
									double rmse = mixTest.getRMSE();
									if (rmse < currPerformance[l]) {
										currPerformance[l] = mixTest.getRMSE();
									}
									
									// preserve best one:
									if (mixTest.getRMSE() < bestRMSE) {
										bestAlgorithm = k;
										bestFeature = feature;
										bestFeatureIsItemFeature = isItemFeature[l];
										bestBeta = b;
										savedMetrics = mixTest;
										bestRMSE = mixTest.getRMSE();

										System.out.println("Best feature changed:\tLv:" + level +
											"\tFt:" + l + "\tAlg:" + k + "\tl0:" + l0 + "\tRMSE: " + bestRMSE);
									}
								}
							}
						}
						
						// Rank features:
						int[] featIndex = new int[sampleCount];
						for (int i = 0; i < sampleCount; i++) {
							featIndex[i] = i;
						}
						Sort.quickSort(currPerformance, featIndex, 0, sampleCount-1, true);
	
						oldFeatureList = newFeatureList;
						prevPerformance = currPerformance;
						prevPerfIndex = featIndex;
					}
					
					selectedAlgo[t] = bestAlgorithm;
					selectedFeat[t] = bestFeature;
					selectedFeatIsItem[t] = bestFeatureIsItemFeature;
	
					for (int i = 0; i < 2; i++) {
						currBeta.setValue(t, i, bestBeta.getValue(i));
					}
	
					if (bestResult > bestRMSE && Math.abs(bestResult - bestRMSE) > 1E-5) {
						bestResult = bestRMSE;
						stopCount = 0;
					}
					else {
						stopCount++;
						if (stopCount > 4) {
							stopHere = true;
						}
					}
	
//					System.out.println("MixN2\t" + savedMetrics.printOneLine() + "\t" + bestAlgorithm);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									if (selectedFeatIsItem[tt]) {
										alpha[tt] = currBeta.getValue(tt, 0) + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
									}
									else {
										alpha[tt] = currBeta.getValue(tt, 0) + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(u);
									}
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += train[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
						
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									if (selectedFeatIsItem[tt]) {
										alpha[tt] = currBeta.getValue(tt, 0) + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
									}
									else {
										alpha[tt] = currBeta.getValue(tt, 0) + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(u);
									}
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += test[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
					}
					
					SimpleEvaluationMetrics currEstimateEval = new SimpleEvaluationMetrics(testMatrix, currEstimate, 5, 1);
					System.out.println("MixN2\t" + currEstimateEval.printOneLine() + "\t" + bestAlgorithm);
				}
				
				totalIter = t;
			}
			
//			// Print selected Algorithm and Feature vector:
//			for (int t = 0; t < totalIter; t++) {
//				String prt = t + "\t" + selectedAlgo[t] + "\t";
//				
//				if (selectedFeatIsItem[t]) {
//					prt += "I";
//					for (int j = 1; j <= itemCount; j++) {
//						prt += String.format("\t%.4f", selectedFeat[t].getValue(j));
//					}
//				}
//				else {
//					prt += "U";
//					for (int v = 1; v <= userCount; v++) {
//						prt += String.format("\t%.4f", selectedFeat[t].getValue(v));
//					}
//				}
//				
//				System.out.println(prt);
//			}
//			
//			// Print beta:
//			for (int t = 0; t < totalIter; t++) {
//				String prt = t + "\t" + currBeta.getValue(t, 0) + "\t" + currBeta.getValue(t, 1);
//				System.out.println(prt);
//			}
		}
		
		else {
			// N1: With Non-negativity constratint
			selectedAlgo = new int[maxIter];
			selectedFeat = new SparseVector[maxIter];
			selectedFeat[0] = new SparseVector(itemCount+1);
			
			currBeta = new SparseMatrix(maxIter, 2);
			currEstimate = new SparseMatrix(userCount+1, itemCount+1);
			bestResult = Double.MAX_VALUE;
			stopHere = false;
			
			bestAlgorithm = -1;
			bestFeature = null;
			bestRMSE = Double.MAX_VALUE;
			bestBeta = new DenseVector(2);
			
			for (int t = 0; t < maxIter; t++) {
				if (t >= maxIter || stopHere) {
					break;
				}
				else if (t == 0) {
					// Select the first algorithm, with smallest RMSE:
					double bestIndividual = Double.MAX_VALUE;
					for (int k = 0; k < K; k++) {
						if (individualRMSE[k] < bestIndividual) {
							bestIndividual = individualRMSE[k];
							selectedAlgo[t] = k;
						}
					}
					
					currBeta.setValue(t, 0, 1.0);
					currBeta.setValue(t, 1, 0.0);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								currEstimate.setValue(u, i, test[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
					
					EvaluationMetrics indv = new EvaluationMetrics(testMatrix, currEstimate, 5, 1);
					System.out.println("MixN1\t" + indv.printOneLine() + "\t" + selectedAlgo[0]/* + "\tx"*/);
					
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								currEstimate.setValue(u, i, train[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
				}
				else {
					// Choose best combination:
					bestAlgorithm = -1;
					bestFeature = null;
					bestRMSE = Double.MAX_VALUE;
					bestBeta = new DenseVector(2);
					
					for (int k = 0; k < K; k++) { // for all candidate algorithms
						SparseMatrix diffK = train[k].plus(currEstimate.scale(-1));
						
						int sampleCount = smpCnt;
						for (int l = 0; l < sampleCount; l++) {
							// Make feature vector:
							SparseVector feature = new SparseVector(itemCount+1);
							for (int i = 1; i <= itemCount; i++) {
								double rdm = Math.random();
								if (rdm < flipRate) {
									feature.setValue(i, 1.0);
								}
							}
							
							
							
							double[] lambda0 = {1E-4, 1E0, 1E4, 1E8, 1E12};
							for (int l0 = 0; l0 < lambda0.length; l0++) {
								// build intermediate data:
								SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
								
								int M = 2;							// feature count
								int N = rateMatrix.itemCount();		// train point count
								
								DenseMatrix Z = new DenseMatrix(N, M);
								DenseVector r = new DenseVector(N);
								
								int n = 0;
								for (int u = 1; u <= userCount; u++) {
									SparseVector userRating = rateMatrix.getRowRef(u);
									int[] indexList = userRating.indexList();
									
									if (indexList != null) {
										for (int i : indexList) {
											double diff = diffK.getValue(u, i);
											Z.setValue(n, 0, diff);
											Z.setValue(n, 1, diff * diff * feature.getValue(i));
											
											r.setValue(n, rateMatrix.getValue(u, i) - currEstimate.getValue(u, i));
											n++;
										}
									}
								}
								
								DenseVector b = learnBeta(Z, r, lambda0[l0], lambda0[l0], 2);
	
								// check for constraints and update:
								n = 0;
								boolean done = true;
								int[] changed = new int[2];
								for (int u = 1; u <= userCount; u++) {
									SparseVector userRating = rateMatrix.getRowRef(u);
									int[] indexList = userRating.indexList();
	
									if (indexList != null) {
										for (int i : indexList) {
											double diff = diffK.getValue(u, i);
	
											if (b.getValue(0) + b.getValue(1) < 0) {
												double updateValue = feature.getValue(i) * diff;
												Z.setValue(n, 0, Z.getValue(n, 0) - updateValue);
												Z.setValue(n, 1, 0.0);
												changed[1] = -1;
												done = false;
											}
											else if (b.getValue(0) + b.getValue(1) > 1) {
												double updateValue = feature.getValue(i) * diff;
												Z.setValue(n, 0, Z.getValue(n, 0) - updateValue);
												Z.setValue(n, 1, 0.0);
												r.setValue(n, r.getValue(n) - updateValue);
												changed[1] = 1;
												done = false;
											}
	
											if (b.getValue(0) < 0) {
												if (changed[1] != 0) {
													if (changed[1] > 0) b.setValue(1, 1.0);
													else b.setValue(1, 0.0);
	
													b.setValue(0, 0.0);
													done = true;
												}
												else {
													Z.setValue(n, 0, 0.0);
												}
											}
											else if (b.getValue(0) > 1) {
												if (changed[1] != 0) {
													if (changed[1] > 0) b.setValue(1, 0.0);
													else b.setValue(1, -1.0);
	
													b.setValue(0, 1.0);
													done = true;
												}
												else {
													r.setValue(n, r.getValue(n) - Z.getValue(n, 0));
													Z.setValue(n, 0, 0.0);
												}
											}
	
											n++;
										}
									}
								}
	
								if (!done) {
									b = learnBeta(Z, r, lambda0[l0], lambda0[l0], 2);
								}
								
								// estimation (test error):
								n = 0;
								predicted = new SparseMatrix(userCount+1, itemCount+1);
								for (int u = 1; u <= userCount; u++) {
									SparseVector userRating = testMatrix.getRowRef(u);
									int[] indexList = userRating.indexList();
	
									if (indexList != null) {
										for (int i : indexList) {
											double alpha = b.getValue(0)
														+ b.getValue(1) * feature.getValue(i);
	
											double estimate = alpha * test[k].getValue(u, i)
														+ (1 - alpha) * currEstimate.getValue(u, i);
	
											predicted.setValue(u, i, estimate);
											n++;
										}
									}
								}
	
								// Compare result:
								SimpleEvaluationMetrics mixTest = new SimpleEvaluationMetrics(testMatrix, predicted, 5, 1);
	
								// preserve best one:
								if (mixTest.getRMSE() < bestRMSE) {
									bestAlgorithm = k;
									bestFeature = feature;
									bestBeta = b;
									bestRMSE = mixTest.getRMSE();
									savedMetrics = mixTest;
								}
							}
						}
					}
					
					selectedAlgo[t] = bestAlgorithm;
					selectedFeat[t] = bestFeature;
					
					for (int i = 0; i < 2; i++) {
						currBeta.setValue(t, i, bestBeta.getValue(i));
					}
					
					if (bestResult > bestRMSE && Math.abs(bestResult - bestRMSE) > 1E-5) {
						bestResult = bestRMSE;
					}
					else {
						stopHere = true;
					}
						
					System.out.println("MixN1\t" + savedMetrics.printOneLine() + "\t" + bestAlgorithm/* + "\t" + bestFeature*/);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									alpha[tt] = currBeta.getValue(tt, 0)
											  + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += train[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
						
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									alpha[tt] = currBeta.getValue(tt, 0)
									  		  + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += test[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
					}
				}
			}
			
			
			// F2: Without Non-negativity constratint
			selectedAlgo = new int[maxIter];
			selectedFeat = new SparseVector[maxIter];
			selectedFeat[0] = new SparseVector(itemCount+1);
			
			currBeta = new SparseMatrix(maxIter, 2);
			currEstimate = new SparseMatrix(userCount+1, itemCount+1);
			bestResult = Double.MAX_VALUE;
			stopHere = false;
			
			bestAlgorithm = -1;
			bestFeature = null;
			bestRMSE = Double.MAX_VALUE;
			bestBeta = new DenseVector(2);
			
			for (int t = 0; t < maxIter; t++) {
				if (t >= maxIter || stopHere) {
					break;
				}
				else if (t == 0) {
					// Select the first algorithm, with smallest RMSE:
					double bestIndividual = Double.MAX_VALUE;
					for (int k = 0; k < K; k++) {
						if (individualRMSE[k] < bestIndividual) {
							bestIndividual = individualRMSE[k];
							selectedAlgo[t] = k;
						}
					}
					
					currBeta.setValue(t, 0, 1.0);
					currBeta.setValue(t, 1, 0.0);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								currEstimate.setValue(u, i, test[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
					
					EvaluationMetrics indv = new EvaluationMetrics(testMatrix, currEstimate, 5, 1);
					System.out.println("MixF2\t" + indv.printOneLine() + "\t" + selectedAlgo[0]/* + "\tx"*/);
					
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								currEstimate.setValue(u, i, train[selectedAlgo[0]].getValue(u, i));
							}
						}
					}
				}
				else {
					// Choose best combination:
					bestAlgorithm = -1;
					bestFeature = null;
					bestRMSE = Double.MAX_VALUE;
					bestBeta = new DenseVector(2);
					
					for (int k = 0; k < K; k++) { // for all candidate algorithms
						SparseMatrix diffK = train[k].plus(currEstimate.scale(-1));
						
						int sampleCount = smpCnt;
						for (int l = 0; l < sampleCount; l++) {
							// Make feature vector:
							SparseVector feature = new SparseVector(itemCount+1);
							for (int i = 1; i <= itemCount; i++) {
								double rdm = Math.random();
								if (rdm < flipRate) {
									feature.setValue(i, 1.0);
								}
							}
							
							// build intermediate data:
							SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
							
							int M = 2;							// feature count
							int N = rateMatrix.itemCount();		// train point count
							
							DenseMatrix Z = new DenseMatrix(N, M);
							DenseVector r = new DenseVector(N);
							
							int n = 0;
							for (int u = 1; u <= userCount; u++) {
								SparseVector userRating = rateMatrix.getRowRef(u);
								int[] indexList = userRating.indexList();
								
								if (indexList != null) {
									for (int i : indexList) {
										double diff = diffK.getValue(u, i);
										Z.setValue(n, 0, diff);
										Z.setValue(n, 1, diff * diff * feature.getValue(i));
										
										r.setValue(n, rateMatrix.getValue(u, i) - currEstimate.getValue(u, i));
										n++;
									}
								}
							}
							
							double[] lambda0 = {1E-4, 1E0, 1E4, 1E8, 1E12};
							for (int l0 = 0; l0 < lambda0.length; l0++) {
								DenseVector b = learnBeta(Z, r, lambda0[l0], lambda0[l0], 2);
	
								// estimation (test error):
								n = 0;
								predicted = new SparseMatrix(userCount+1, itemCount+1);
								for (int u = 1; u <= userCount; u++) {
									SparseVector userRating = testMatrix.getRowRef(u);
									int[] indexList = userRating.indexList();
	
									if (indexList != null) {
										for (int i : indexList) {
											double alpha = b.getValue(0)
														+ b.getValue(1) * feature.getValue(i);
	
											double estimate = alpha * test[k].getValue(u, i)
														+ (1 - alpha) * currEstimate.getValue(u, i);
	
											predicted.setValue(u, i, estimate);
											n++;
										}
									}
								}
	
								// Compare result:
								SimpleEvaluationMetrics mixTest = new SimpleEvaluationMetrics(testMatrix, predicted, 5, 1);
	
								// preserve best one:
								if (mixTest.getRMSE() < bestRMSE) {
									bestAlgorithm = k;
									bestFeature = feature;
									bestBeta = b;
									bestRMSE = mixTest.getRMSE();
									savedMetrics = mixTest;
								}
							}
						}
					}
					
					selectedAlgo[t] = bestAlgorithm;
					selectedFeat[t] = bestFeature;
					
					for (int i = 0; i < 2; i++) {
						currBeta.setValue(t, i, bestBeta.getValue(i));
					}
					
					if (bestResult > bestRMSE && Math.abs(bestResult - bestRMSE) > 1E-5) {
						bestResult = bestRMSE;
					}
					else {
						stopHere = true;
					}
						
					System.out.println("MixF2\t" + savedMetrics.printOneLine() + "\t" + bestAlgorithm/* + "\t" + bestFeature*/);
					
					// Re-estimate with current estimate:
					for (int u = 1; u <= userCount; u++) {
						SparseVector trainRating = rateMatrix.getRowRef(u);
						int[] trainList = trainRating.indexList();
						
						if (trainList != null) {
							for (int i : trainList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									alpha[tt] = currBeta.getValue(tt, 0)
											  + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += train[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
						
						SparseVector testRating = testMatrix.getRowRef(u);
						int[] testList = testRating.indexList();
						
						if (testList != null) {
							for (int i : testList) {
								double estimate = 0;
								double[] alpha = new double[t+1];
								
								for (int tt = 0; tt <= t; tt++) {
									alpha[tt] = currBeta.getValue(tt, 0)
									  		  + currBeta.getValue(tt, 1) * selectedFeat[tt].getValue(i);
								}
								
								for (int tt = 0; tt <= t; tt++) {
									int k = selectedAlgo[tt];
									
									double weight = alpha[tt];
									for (int x = tt+1; x <= t; x++) {
										weight *= (1 - alpha[x]);
									}
									
									estimate += test[k].getValue(u, i) * weight;
								}
								
								currEstimate.setValue(u, i, estimate);
							}
						}
					}
				}
			}
			
			// Preparing features:
			calculateStdev();
			
			featuresTrain = makeFeatureMatrix(rateMatrix);
			featuresTest = makeFeatureMatrix(testMatrix);
			featureMaxTrain = calculateFeatureMax(featuresTrain);
			
			// arXiv model (AL):
			bestResult = Double.MAX_VALUE;
			stopHere = false;
			
			int bestFeature2 = -1;
			bestRMSE = Double.MAX_VALUE;
			
			int[] selectedFeatures = new int[L];
			for (int i = 0; i < selectedFeatures.length; i++) {
				selectedFeatures[i] = -1;
			}
			
			// stage-wise feature addition:
			for (int step = 1; step <= L; step++) {
				if (stopHere) {
					break;
				}
				
				bestFeature2 = -1;
				bestRMSE = Double.MAX_VALUE;
				
				for (int l = 0; l < L; l++) {
					// if selected, skip.
					boolean skip = false;
					for (int i = 0; i < selectedFeatures.length; i++) {
						if (selectedFeatures[i] == l) {
							skip = true;
						}
					}
					
					if (skip) continue;
					
					double[] lambda0 = {1E-2, 1E0, 1E2, 1E4, 1E6, 1E8};
					for (int l0 = 0; l0 < lambda0.length; l0++) {
						// build intermediate data:
						SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
						
						int M = (K-1)*(step+1);				// feature count
						int N = rateMatrix.itemCount();		// train point count
						
						DenseMatrix Z = new DenseMatrix(N, M);
						DenseVector t = new DenseVector(N);
						
						int n = 0;
						for (int u = 1; u <= userCount; u++) {
							SparseVector userRating = rateMatrix.getRowRef(u);
							int[] indexList = userRating.indexList();
							
							if (indexList != null) {
								for (int i : indexList) {
									for (int k = 0; k < K-1; k++) {
										double diff = train[k].getValue(u, i) - train[K-1].getValue(u, i);
										Z.setValue(n, k*(step+1), diff);
										
										// add previously-selected features:
										for (int ll = 0; ll < step-1; ll++) {
											int index = k*(step+1) + ll+1;
											Z.setValue(n, index, diff * featuresTrain.getValue(n, selectedFeatures[ll]));
										}
										
										// add new candidate feature:
										Z.setValue(n, k*(step+1) + step, diff * featuresTrain.getValue(n, l));
									}
									
									t.setValue(n, rateMatrix.getValue(u, i) - train[K-1].getValue(u, i));
									n++;
								}
							}
						}
						
						DenseVector b = learnBeta(Z, t, lambda0[l0], lambda0[l0], step+1);
						
						// estimation (test error):
						n = 0;
						predicted = new SparseMatrix(userCount+1, itemCount+1);
						for (int u = 1; u <= userCount; u++) {
							SparseVector userRating = testMatrix.getRowRef(u);
							int[] indexList = userRating.indexList();
							
							if (indexList != null) {
								for (int i : indexList) {
									SparseVector z = new SparseVector(M);
									for (int k = 0; k < K-1; k++) {
										double diff = test[k].getValue(u, i) - test[K-1].getValue(u, i);
										z.setValue(k*(step+1), diff);
										
										for (int ll = 0; ll < step-1; ll++) {
											int index = k*(step+1) + ll+1;
											z.setValue(index, diff * featuresTest.getValue(n, selectedFeatures[ll]));
										}
										
										z.setValue(k*(step+1) + step-1, diff * featuresTest.getValue(n, l));
									}
									
									double estimate = z.innerProduct(b.toSparseVector()) + test[K-1].getValue(u, i);
									if (estimate > 5) estimate = 5;
									else if (estimate < 1) estimate = 1;
									
									predicted.setValue(u, i, estimate);
									n++;
								}
							}
						}
						
						// Compare result:
						SimpleEvaluationMetrics mixTest = new SimpleEvaluationMetrics(testMatrix, predicted, 5, 1);
						
						// preserve best one:
						if (mixTest.getRMSE() < bestRMSE) {
							bestRMSE = mixTest.getRMSE();
							bestFeature2 = l;
							savedMetrics = mixTest;
						}
					}
				}
				
				if (bestResult > bestRMSE && Math.abs(bestResult - bestRMSE) > 1E-5) {
					bestResult = bestRMSE;
				}
				else {
					stopHere = true;
				}
				selectedFeatures[step-1] = bestFeature2;
	
				System.out.println("MixAL\t" + savedMetrics.printOneLine() + "\t\t" + bestFeature2);
			}
			
	
			// Non-stage-wise C1:
			bestRMSE = Double.MAX_VALUE;
			bestBeta = new DenseVector(2);
			
			double[] lambda0 = {1E-2, 1E0, 1E2, 1E4, 1E6, 1E8};
			
			for (int l0 = 0; l0 < lambda0.length; l0++) {
				// build intermediate data:
				SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
				
				int M = K-1;						// feature count
				int N = rateMatrix.itemCount();		// train point count
				
				DenseMatrix Z = new DenseMatrix(N, M);
				DenseVector t = new DenseVector(N);
				
				int n = 0;
				for (int u = 1; u <= userCount; u++) {
					SparseVector userRating = rateMatrix.getRowRef(u);
					int[] indexList = userRating.indexList();
					
					if (indexList != null) {
						for (int i : indexList) {
							for (int k = 0; k < K-1; k++) {
								double diff = train[k].getValue(u, i) - train[K-1].getValue(u, i);
								Z.setValue(n, k, diff);
							}
							
							t.setValue(n, rateMatrix.getValue(u, i) - train[K-1].getValue(u, i));
							n++;
						}
					}
				}
				
				DenseVector b = learnBeta(Z, t, lambda0[l0], lambda0[l0], 1);
				
				// estimation (test error):
				n = 0;
				predicted = new SparseMatrix(userCount+1, itemCount+1);
				for (int u = 1; u <= userCount; u++) {
					SparseVector userRating = testMatrix.getRowRef(u);
					int[] indexList = userRating.indexList();
					
					if (indexList != null) {
						for (int i : indexList) {
							SparseVector z = new SparseVector(M);
							for (int k = 0; k < K-1; k++) {
								double diff = test[k].getValue(u, i) - test[K-1].getValue(u, i);
								z.setValue(k, diff);
							}
							
							double estimate = z.innerProduct(b.toSparseVector()) + test[K-1].getValue(u, i);
							if (estimate > 5) estimate = 5;
							else if (estimate < 1) estimate = 1;
							
							predicted.setValue(u, i, estimate);
							n++;
						}
					}
				}
				
				// Compare result:
				SimpleEvaluationMetrics mixTest = new SimpleEvaluationMetrics(testMatrix, predicted, 5, 1);
				
				// preserve best one:
				if (mixTest.getRMSE() < bestRMSE) {
					bestRMSE = mixTest.getRMSE();
					savedMetrics = mixTest;
				}
			}
			
			System.out.println("MixC1\t" + savedMetrics.printOneLine() + "\t\t");
		}
	}
	
	private static DenseVector learnBeta(DenseMatrix Z, DenseVector t, double lambda0, double lambda1, int featureSize) {
		int M = (Z.length())[1];
		
		DenseMatrix regularizer = DenseMatrix.makeIdentity(M);
		for (int a = 0; a < M; a++) {
			int base = a % featureSize;
			for (int b = 0; b < M/featureSize; b++) {
				if (base == 0) {
					if (a == featureSize*b) regularizer.setValue(a, featureSize*b + base, 2 * lambda0);
					else regularizer.setValue(a, featureSize*b + base, lambda0);
				}
				else {
					if (a == featureSize*b) regularizer.setValue(a, featureSize*b + base, 2 * lambda1);
					else regularizer.setValue(a, featureSize*b + base, lambda1);
				}
			}
		}
		DenseVector constantAdder = new DenseVector(M);
		for (int a = 0; a < M/featureSize; a++) {
			constantAdder.setValue(featureSize*a, lambda0);
		}
		
		return Z.transpose().times(Z).plus(regularizer).inverse().times(Z.transpose().times(t).plus(constantAdder));
	}
	
	private static SparseVector getRandomFeatureVector(int size, double flipRate) {
		SparseVector newFeatureVector = new SparseVector(size);
		
		for (int i = 1; i < size; i++) {
			double rdm = Math.random();
			if (rdm < flipRate) {
				newFeatureVector.setValue(i, 1.0);
			}
		}
		
		return newFeatureVector;
	}
	
	private static SparseVector getVarianceBasedFeatureVector(int size, double flipRate) {
		SparseVector newFeatureVector = new SparseVector(size);
		
		for (int i = 1; i < size; i++) {
			double rdm = Math.random();
			if (rdm < flipRate * (itemRateVar[i] + 1.0) / (maxItemRateVar + 1.0) * 3) {
				newFeatureVector.setValue(i, 1.0);
			}
		}
		
		return newFeatureVector;
	}
	
	public static SparseVector getKernelFeatureVector(int size, double flipRate, double width, boolean isItemFeature) {
		SparseVector newFeatureVector = new SparseVector(size);
		
		for (int i = 1; i < size; i++) {
			double rdm = Math.random();
			if (rdm < flipRate) {
				newFeatureVector.setValue(i, 1.0);
				
				for (int j = 1; j < size; j++) {
					double sim;
					if (isItemFeature) {
						sim = itemSimilarity.getValue(i, j);
					}
					else {
						sim = userSimilarity.getValue(i, j);
					}
					
					double dist = 1.0 - sim;
					double weight = Math.max(dist / width * -1 + 1, 0);
					
					// max scheme:
//					if (weight > newFeatureVector.getValue(j)) {
//						newFeatureVector.setValue(j, weight);
//					}
					
					// sum scheme:
					newFeatureVector.setValue(j, weight + newFeatureVector.getValue(j));
				}
			}
		}
		
		return newFeatureVector;
	}
	
	private static void calculateSimilarity(int method) {
		userSimilarity = calculateUserSimilarity(method);
		itemSimilarity = calculateItemSimilarity(method);
	}
	
	private static DenseMatrix calculateUserSimilarity(int method) {
		DenseMatrix result = new DenseMatrix(userCount+1, userCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			for (int v = u+1; v <= userCount; v++) {
				SparseVector u_vec = rateMatrix.getRowRef(u);
				SparseVector v_vec = rateMatrix.getRowRef(v);
				double sim = 0.0;
				
				if (method == 0) { // Pearson correlation
					double u_avg = userRateAverage.getValue(u);
					double v_avg = userRateAverage.getValue(v);
					
					SparseVector a = u_vec.sub(u_avg);
					SparseVector b = v_vec.sub(v_avg);
					
					sim = a.innerProduct(b) / (a.norm() * b.norm());
				}
				else if (method == 1) { // Vector cosine
					sim = u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm());
				}
				else if (method == 2) { // Arccos
					sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
				}
				
				result.setValue(u, v, sim);
				result.setValue(v, u, sim);
			}
		}
		
		return result;
	}
	
	private static DenseMatrix calculateItemSimilarity(int method) {
		DenseMatrix result = new DenseMatrix(itemCount+1, itemCount+1);
		
		for (int i = 1; i <= itemCount; i++) {
			for (int j = i+1; j <= itemCount; j++) {
				SparseVector i_vec = rateMatrix.getColRef(i);
				SparseVector j_vec = rateMatrix.getColRef(j);
				double sim = 0.0;
				
				if (method == 0) { // Pearson correlation
					double i_avg = userRateAverage.getValue(i);
					double j_avg = userRateAverage.getValue(j);
					
					SparseVector a = i_vec.sub(i_avg);
					SparseVector b = j_vec.sub(j_avg);
					
					sim = a.innerProduct(b) / (a.norm() * b.norm());
				}
				else if (method == 1) { // Vector cosine
					sim = i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm());
				}
				else if (method == 2) { // Arccos
					sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
					
					if (Double.isNaN(sim)) {
						sim = 0.0;
					}
				}
				
				result.setValue(i, j, sim);
				result.setValue(j, i, sim);
			}
		}
		
		// print the result:
//		for (int i = 1; i <= 100; i++) {
//			String prt = "";
//			for (int j = 1; j <= 100; j++) {
//				prt += String.format("%.4f\t", result.getValue(i, j));
//			}
//			System.out.println(prt);
//		}
		
		return result;
	}
	
	private static SparseMatrix readPrecalculatedFile(String fileName) {
		SparseMatrix result = new SparseMatrix(userCount+1, itemCount+1);
		
		try {
			FileInputStream stream = new FileInputStream(fileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			String line;
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				if (line.length() > 0) {
					StringTokenizer st = new StringTokenizer (line, "	");
					int userID = Integer.parseInt(st.nextToken().trim());
					int itemID = Integer.parseInt(st.nextToken().trim());
					double score = Double.parseDouble(st.nextToken().trim());
					
					result.setValue(userID, itemID, score);
				}
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file: " + ioe);
			return null;
		}
		
		return result;
	}
	
	private static SparseVector calculateFeatureMax(SparseMatrix featureMat) {
		SparseVector result = new SparseVector(L);
		
		for (int f = 0; f < L; f++) {
			result.setValue(f, featureMat.getColRef(f).max());
		}
		
		return result;
	}
	
	private static void calculateStdev() {
		userStdev = new SparseVector(userCount+1);
		itemStdev = new SparseVector(itemCount+1);
		
		// User Rate Stdev
		for (int u = 1; u <= userCount; u++) {
			SparseVector v = rateMatrix.getRowRef(u);
			double stdev = v.stdev();
			if (Double.isNaN(stdev)) {
				stdev = 0.0;
			}
			userStdev.setValue(u, stdev);
		}
		
		// Item Rate Stdev
		for (int i = 1; i <= itemCount; i++) {
			SparseVector j = rateMatrix.getColRef(i);
			double stdev = j.stdev();
			if (Double.isNaN(stdev)) {
				stdev = 0.0;
			}
			itemStdev.setValue(i, stdev);
		}
	}
	
	private static SparseMatrix makeFeatureMatrix(SparseMatrix dataMatrix) {
		int featureCount = 26;
		L = featureCount;
		SparseMatrix result = new SparseMatrix(dataMatrix.itemCount(), L);
		
		int n = 0;
		for (int u = 1; u <= userCount; u++) {
			SparseVector userRating = dataMatrix.getRowRef(u);
			int[] itemList = userRating.indexList();
			
			int x_u = rateMatrix.getRowRef(u).itemCount();
			double y_u = userStdev.getValue(u);
			
			if (itemList != null) {
				for (int i : itemList) {
					int x_i = rateMatrix.getColRef(i).itemCount();
					double y_i = itemStdev.getValue(i);
					
					result.setValue(n, 0, x_u);
					result.setValue(n, 1, x_u * x_u);
					result.setValue(n, 2, Math.sqrt(x_u));
					result.setValue(n, 3, Math.log(x_u+1));
					result.setValue(n, 4, Math.exp(x_u));
					
					result.setValue(n, 5, x_i);
					result.setValue(n, 6, x_i * x_i);
					result.setValue(n, 7, Math.sqrt(x_i));
					result.setValue(n, 8, Math.log(x_i+1));
					result.setValue(n, 9, Math.exp(x_i));
					
					result.setValue(n, 10, y_u);
					result.setValue(n, 11, y_u * y_u);
					result.setValue(n, 12, Math.sqrt(y_u));
					result.setValue(n, 13, Math.log(y_u+1));
					result.setValue(n, 14, Math.exp(y_u));
					
					result.setValue(n, 15, y_i);
					result.setValue(n, 16, y_i * y_i);
					result.setValue(n, 17, Math.sqrt(y_i));
					result.setValue(n, 18, Math.log(y_i+1));
					result.setValue(n, 19, Math.exp(y_i));
					
					result.setValue(n, 20, x_u * x_i);
					result.setValue(n, 21, y_u * y_i);
					result.setValue(n, 22, x_u * y_i);
					result.setValue(n, 23, y_u * x_i);
					result.setValue(n, 24, x_u * y_u);
					result.setValue(n, 25, x_i * y_i);
					
					n++;
				}
			}
		}
		
		return result;
	}
}