

import prea.recommender.llorma.WeakLearner;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

import prea.data.splitter.*;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.*;
import prea.recommender.matrix.RegularizedSVD;
import prea.util.EvaluationMetrics;
import prea.util.Printer;

/**
 * A main class for ensemble experiments.
 * 
 * @author Joonseok Lee
 * @since 2012. 8. 22
 * @version 1.1
 */
public class LLRMA {
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
	public final static int TRIANGULAR_KERNEL = 201;
	public final static int UNIFORM_KERNEL = 202;
	public final static int EPANECHNIKOV_KERNEL = 203;
	public final static int GAUSSIAN_KERNEL = 204;
	
	/**
	 * Test examples for every algorithm. Also includes parsing the given parameters.
	 * 
	 * @param argv The argument list. Each element is separated by an empty space.
	 * First element is the data file name, and second one is the algorithm name.
	 * Third and later includes parameters for the chosen algorithm.
	 * Please refer to our web site for detailed syntax.
	 * @throws InterruptedException 
	 */
	public static void main(String argv[]) throws InterruptedException {
		// Set default setting first:
		dataFileName = "yelp";
		evaluationMode = DataSplitManager.SIMPLE_SPLIT;
		splitFileName = dataFileName + "_split.txt";
		testRatio = 0.2;
		foldCount = 8;
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
	
	/** Run an/all algorithm with given data, based on the setting from command arguments. 
	 * @throws InterruptedException */
	private static void run() throws InterruptedException {
		System.out.println(EvaluationMetrics.printTitle() + "\tTrain Time\tTest Time");
		
		int maxIter, rank, modelMax;
		double kernelWidth;
		
		//System.out.println("[START] Regularized SVD");
		RegularizedSVD regsvd = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 1, 0.005, 0.01, 0, 100, false);
		System.out.println(testRecommender("SVD-1", regsvd));
		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 3, 0.005, 0.008, 0, 100, false);
		System.out.println(testRecommender("SVD-3", baseline));
		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 5, 0.005, 0.006, 0, 100, false);
		System.out.println(testRecommender("SVD-5", baseline));
		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 7, 0.005, 0.004, 0, 100, false);
		System.out.println(testRecommender("SVD-7", baseline));
		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 10, 0.005, 0.002, 0, 100, false);
		System.out.println(testRecommender("SVD-10", baseline));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 15, 0.005, 0.1, 0, 200, false);
//		System.out.println(testRecommender("SVD-15", baseline));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 20, 0.005, 0.1, 0, 200, false);
//		System.out.println(testRecommender("SVD-20", baseline));
		
		// best parameters for Bookcrossing (0.005���� ������ ���� ���غ���.)
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 1, 0.005, 0.5, 0, 200, false);
//		System.out.println(testRecommender("SVD-1", regsvd));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 3, 0.005, 0.5, 0, 200, false);
//		System.out.println(testRecommender("SVD-3", baseline));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 5, 0.005, 0.5, 0, 200, false);
//		System.out.println(testRecommender("SVD-5", baseline));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 7, 0.005, 0.5, 0, 200, false);
//		System.out.println(testRecommender("SVD-7", baseline));
//		baseline = new RegularizedSVD(userCount, itemCount, maxValue, minValue, 10, 0.005, 0.5, 0, 200, false);
//		System.out.println(testRecommender("SVD-10", baseline));
		
//		baseline = regsvd;
		
		
		//System.out.println("[START] Calculating similarity");
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
		
		
		// Global combination
		modelMax = 50; maxIter = 100; kernelWidth = 0.8; rank = 1;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		globalCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 50; maxIter = 100; kernelWidth = 0.8; rank = 3;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		globalCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 50; maxIter = 100; kernelWidth = 0.8; rank = 5;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		globalCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 50; maxIter = 100; kernelWidth = 0.8; rank = 7;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		globalCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 50; maxIter = 100; kernelWidth = 0.8; rank = 10;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		globalCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		
		// Parallel combination
		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 1;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 3;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 5;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 7;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
		
		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 10;
		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
		System.out.println();
//
//		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 15;
//		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
//		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
//		System.out.println();
//		
//		modelMax = 55; maxIter = 100; kernelWidth = 0.8; rank = 20;
//		System.out.println("Kernel=EPANECHNIKOV, width=" + kernelWidth + ", rank=" + rank);
//		parallelCombination(rank, Math.min(testMatrix.itemCount(), modelMax), maxIter, EPANECHNIKOV_KERNEL, kernelWidth);
//		System.out.println();
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
	
	public static void globalCombination(int rank, int modelMax, int maxIter, int kernelType, double kernelWidth) {
		SparseMatrix[] localUserFeatures = new SparseMatrix[modelMax];
		SparseMatrix[] localItemFeatures = new SparseMatrix[modelMax];
		EvaluationMetrics preservedEvalMet = null;
		
		long learnStart = System.currentTimeMillis();
		
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
					weightSum[u][i] += kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 * kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType);
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
		
		double learningRate = 0.2;
		double regularizer = 0.001;
		boolean showProgress = false;
		
		while (Math.abs(prevErr - currErr) > 0.0001 && round < maxIter) {
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						// current estimation:
						double RuiEst = 0.0;
						for (int l = 0; l < modelMax; l++) {
							RuiEst += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
									* kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	* kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
								 	/ weightSum[u][i];
						}
						double RuiReal = rateMatrix.getValue(u, i);
						double err = RuiReal - RuiEst;
						
						// parameter update:
						for (int l = 0; l < modelMax; l++) {
							double weight = kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
									 	  * kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
									 	  / weightSum[u][i];
							
							for (int r = 0; r < rank; r++) {
								double Fus = localUserFeatures[l].getValue(u, r);
								double Gis = localItemFeatures[l].getValue(r, i);
								
								localUserFeatures[l].setValue(u, r, Fus + learningRate*(err*Gis*weight - regularizer*Fus)); // CHECK REGULARIZATION!
								if(Double.isNaN(Fus + learningRate*(err*Gis*weight - regularizer*Fus))) {
									System.out.println("a");
								}
								localItemFeatures[l].setValue(r, i, Gis + learningRate*(err*Fus*weight - regularizer*Gis));
								if(Double.isNaN(Gis + learningRate*(err*Fus*weight - regularizer*Gis))) {
									System.out.println("b");
								}
							}
						}
					}
				}
			}
			
			
			// Test:
			SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
			
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = testMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						double prediction = 0.0;
						for (int l = 0; l < modelMax; l++) {
							prediction += localUserFeatures[l].getRow(u).innerProduct(localItemFeatures[l].getCol(i))
									* kernelize(getUserSimilarity(anchorUser[l], u), kernelWidth, kernelType)
								 	* kernelize(getItemSimilarity(anchorItem[l], i), kernelWidth, kernelType)
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
			
			EvaluationMetrics e = new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
			System.out.println(round + "\t" + e.printOneLine());
			preservedEvalMet = e;
			
			prevErr = currErr;
			currErr = e.getRMSE();//sum/rateCount;
			
			round++;
			
			// Show progress:
			if (showProgress) {
				System.out.println(round + "\t" + currErr);
			}
		}
		
		long learnEnd = System.currentTimeMillis();
		
		System.out.println("GLB-" + rank + "\t"
			+ preservedEvalMet.printOneLine() + "\t"
			+ Printer.printTime(learnEnd - learnStart) + "\t"
			+ Printer.printTime(0));
	}
	
	public static void parallelCombination(int rank, int modelMax, int maxIter, int kernelType, double kernelWidth) throws InterruptedException {
		final int RUNNING_THREAD_MAX = 8;
		
		int completeModelCount = 0;
		long learnStart = System.currentTimeMillis();
		
		WeakLearner[] learners = new WeakLearner[RUNNING_THREAD_MAX];
		int[] anchorUser = new int[modelMax];
		int[] anchorItem = new int[modelMax];
		
		int modelCount = 0;
		double anchorErrCum = 0.0;
		int[] runningThreadList = new int[RUNNING_THREAD_MAX];
		int runningThreadCount = 0;
		int waitingThreadPointer = 0;
		int nextRunningSlot = 0;
		
		double anchorErr = 0.0;
		double lowestRMSE = Double.MAX_VALUE;
		EvaluationMetrics preservedEvalMet = null;
		
		SparseMatrix cumPrediction = new SparseMatrix(userCount+1, itemCount+1);
		SparseMatrix cumWeight = new SparseMatrix(userCount+1, itemCount+1);
		
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

					learners[nextRunningSlot] = new WeakLearner(modelCount, rank, userCount, itemCount, u_t, i_t, 0.01, 0.001, 100, w, v, rateMatrix);
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
					
					SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
					for (int u = 1; u <= userCount; u++) {
						int[] testItems = testMatrix.getRowRef(u).indexList();
						
						if (testItems != null) {
							for (int t = 0; t < testItems.length; t++) {
								int i = testItems[t];
								
								double weight = kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType)
												* kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType);
								double newPrediction = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(i)) * weight;
								cumWeight.setValue(u, i, cumWeight.getValue(u, i) + weight);
								cumPrediction.setValue(u, i, cumPrediction.getValue(u, i) + newPrediction);
								double prediction = cumPrediction.getValue(u, i) / cumWeight.getValue(u, i);
								
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
								
								if (u == anchorUser[mc] && i == anchorItem[mc]) {
									anchorErr = Math.abs(prediction - testMatrix.getValue(u, i));
									anchorErrCum += Math.pow(anchorErr, 2);
								}
							}
						}
					}
					
					EvaluationMetrics e = new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
					System.out.println((modelCount - RUNNING_THREAD_MAX + 1) + "\t" + learners[mp].getAnchorUser() + "\t" + learners[mp].getAnchorItem() + "\t"
						+ learners[mp].getTrainErr() + "\t" + e.getRMSE() + "\t" + anchorErr + "\t" + Math.sqrt(anchorErrCum/(mp+1)));
					
					if (e.getRMSE() < lowestRMSE) {
						lowestRMSE = e.getRMSE();
						preservedEvalMet = e;
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
		
		System.out.println("PRL-" + rank + "\t"
			+ preservedEvalMet.printOneLine() + "\t"
			+ Printer.printTime(learnEnd - learnStart) + "\t"
			+ Printer.printTime(0));
	}
	
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
	
	public static double kernelize(double sim, double width, int kernelType) {
		double dist = 1.0 - sim;
		
		if (kernelType == TRIANGULAR_KERNEL) { // Triangular kernel
			return Math.max(1 - dist/width, 0);
		}
		else if (kernelType == UNIFORM_KERNEL) {
			return dist < width ? 1 : 0;
		}
		else if (kernelType == EPANECHNIKOV_KERNEL) {
			return Math.max(3.0/4.0 * (1 - Math.pow(dist/width, 2)), 0);
		}
		else if (kernelType == GAUSSIAN_KERNEL) {
			return 1/Math.sqrt(2*Math.PI) * Math.exp(-0.5 * Math.pow(dist/width, 2));
		}
		else { // Default: Triangular kernel
			return Math.max(1 - dist/width, 0);
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
			
			newFeatureVector.setValue(i, kernelize(sim, width, kernelType));
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
		
		return result;
	}
}