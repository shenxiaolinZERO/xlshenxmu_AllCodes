package prea.data.splitter;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import prea.data.structure.SparseMatrix;

/**
 * This class helps to split data matrix into train set and test set,
 * based on the test set ratio defined by the user.
 * (This file is a special version for rank-based loss experiment.)
 * 
 * @author Joonseok Lee
 * @since 2013. 4. 23
 * @version 1.1
 */
public class SimpleSplit2 extends DataSplitManager {
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct an instance for simple splitter. */
	public SimpleSplit2(SparseMatrix originalMatrix, double testRatio, int max, int min) {
		super(originalMatrix, max, min);
		split(testRatio);
		calculateAverage((maxValue + minValue) / 2);
	}
	
	/**
	 * Items which will be used for test purpose are moved from rateMatrix to testMatrix.
	 * 
	 * @param testRatio proportion of items which will be used for test purpose. 
	 *  
	 */
	private void split(double testRatio) {
		if (testRatio > 1 || testRatio < 0) {
			return;
		}
		else {
			recoverTestItems();
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					for (int i : itemList) {
						double rdm = Math.random();
						
//						// Learn on Train set, Test on whole set
//						testMatrix.setValue(u, i, rateMatrix.getValue(u, i));
//						if (rdm < testRatio) {
//							rateMatrix.setValue(u, i, 0.0);
//						}
//						
//						// Learn on Train set, Test on Train set
//						if (rdm < testRatio) {
//							rateMatrix.setValue(u, i, 0.0);
//						}
//						testMatrix = rateMatrix;
//						
						// Learn on Train set, Test on Test set
						if (rdm < testRatio) {
							testMatrix.setValue(u, i, rateMatrix.getValue(u, i));
							rateMatrix.setValue(u, i, 0.0);
						}
					}
				}
			}
		}
/*		
try {
	FileOutputStream outputStream = new FileOutputStream("MovieLens100K_train_simple.lsvm");
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
	
	FileOutputStream outputStream2 = new FileOutputStream("MovieLens100K_test_simple.lsvm");
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
} catch(IOException e){}
*/
	}
}
