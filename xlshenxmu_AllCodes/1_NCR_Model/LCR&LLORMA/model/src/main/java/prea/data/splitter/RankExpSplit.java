package prea.data.splitter;

import prea.data.structure.SparseMatrix;
import prea.util.Sort;
import java.io.*;

/**
 * This class helps to split data matrix into train set and test set,
 * with constant number of training examples for each user.
 * Users with less than the constant are simply dropped.
 * 
 * @author Joonseok Lee
 * @since 2013. 8. 10
 * @version 1.2
 */
public class RankExpSplit extends DataSplitManager {
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct an instance for simple splitter. */
	public RankExpSplit(SparseMatrix originalMatrix, int userTrainCount, int minTestCount, int max, int min) {
		super(originalMatrix, max, min);
		split(userTrainCount, minTestCount);
		calculateAverage((maxValue + minValue) / 2);
	}
	
	/**
	 * Items which will be used for test purpose are moved from rateMatrix to testMatrix.
	 * 
	 * @param userTrainCount The number of training items for each user.
	 * @param minTestCount The number of test items guaranteed for each user.
	 */
	private void split(int userTrainCount, int minTestCount) {
		//*****************************************************************************
		int VALIDATION_COUNT = 10;
		//*****************************************************************************
		
		if (userTrainCount <= 0) {
			return;
		}
		else {
			recoverTestItems();
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList.length >= userTrainCount + minTestCount) {
					double[] rdmList = new double[itemList.length];

					for (int t = 0; t < rdmList.length; t++) {
						rdmList[t] = Math.random();
					}
					
					Sort.kLargest(rdmList, itemList, 0, itemList.length - 1, userTrainCount + VALIDATION_COUNT);

					// (Randomly-chosen) first N items remains in rateMatrix.
					// Rest of them are moved to testMatrix.
					for (int t = userTrainCount; t < itemList.length; t++) {
						testMatrix.setValue(u, itemList[t], rateMatrix.getValue(u, itemList[t]));
						rateMatrix.setValue(u, itemList[t], 0.0);
					}
					
//					//*****************************************************************************
//					// DROP 10 ITEMS FROM TEST SET
//					int validCount = Math.min(VALIDATION_COUNT, itemList.length - userTrainCount - minTestCount);
//					for (int t = userTrainCount; t < userTrainCount + validCount; t++) {
//						rateMatrix.setValue(u, itemList[t], 0.0);
//					}
//					for (int t = userTrainCount + validCount; t < itemList.length; t++) {
//						testMatrix.setValue(u, itemList[t], rateMatrix.getValue(u, itemList[t]));
//						rateMatrix.setValue(u, itemList[t], 0.0);
//					}
//					//*****************************************************************************
				}
				else { // drop the user both from train/test matrix
					for (int t = 0; t < itemList.length; t++) {
						testMatrix.setValue(u, itemList[t], 0.0);
						rateMatrix.setValue(u, itemList[t], 0.0);
					}
				}
			}
		}
		
/*
// Printing svmlib type train/test split data. (used for CofiRank)
try {
FileOutputStream outputStream = new FileOutputStream("EachMovie_train" + userTrainCount + ".lsvm");
PrintWriter pSystemTrain = new PrintWriter (outputStream);
for (int u = 1; u <= userCount; u++) {
	if (rateMatrix.getRowRef(u).itemCount() + testMatrix.getRowRef(u).itemCount() > 0) {
		String tmp = "";
		int[] itemList = rateMatrix.getRowRef(u).indexList();
		for (int i : itemList) {
			tmp += (i + ":" + (int) rateMatrix.getValue(u, i) + " ");
		}
		pSystemTrain.println(tmp);
	}
}
pSystemTrain.flush();
outputStream.close();

FileOutputStream outputStream2 = new FileOutputStream("EachMovie_test" + userTrainCount + ".lsvm");
PrintWriter pSystemTest = new PrintWriter (outputStream2);
for (int u = 1; u <= userCount; u++) {
	if (rateMatrix.getRowRef(u).itemCount() + testMatrix.getRowRef(u).itemCount() > 0) {
		String tmp = "";
		int[] itemList = testMatrix.getRowRef(u).indexList();
		for (int i : itemList) {
			tmp += (i + ":" + (int) testMatrix.getValue(u, i) + " ");
		}
		pSystemTest.println(tmp);
	}
}
pSystemTest.flush();
outputStream2.close();

FileOutputStream outputStream3 = new FileOutputStream("EachMovie_full" + userTrainCount + ".lsvm");
PrintWriter pSystemFull = new PrintWriter (outputStream3);
for (int u = 1; u <= userCount; u++) {
	if (rateMatrix.getRowRef(u).itemCount() + testMatrix.getRowRef(u).itemCount() > 0) {
		String tmp = "";
		int[] itemList = (rateMatrix.getRow(u).plus(testMatrix.getRow(u))).indexList();
		for (int i : itemList) {
			tmp += (i + ":" + (int) (rateMatrix.getValue(u, i) + testMatrix.getValue(u, i)) + " ");
		}
		pSystemFull.println(tmp);
	}
}
pSystemFull.flush();
outputStream3.close();
} catch(IOException e){}
*/
		
		System.out.println(rateMatrix.itemCount() + "\t" + testMatrix.itemCount());
	}
}
