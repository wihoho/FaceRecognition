package Validation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;
import Training.*;

public class PerformanceTest {
	public static void main(String args[]){
		//System.out.print("LPP: ");
//		test(2,0.5,0,3,2);
//		test(2,0.6,0,3,2);
//		test(2,0.7,0,3,2);
//		test(2,0.8,0,3,2);
//		test(2,0.9,0,3,2);
		
		test(2,0.91,0,3,2);
		test(2,0.92,0,3,2);
		test(2,0.93,0,3,2);
		test(2,0.94,0,3,2);
		test(2,0.95,0,3,2);
		test(2,0.96,0,3,2);
		test(2,0.97,0,3,2);
		test(2,0.98,0,3,2);
		test(2,0.99,0,3,2);
		test(2,1,0,3,2);
	}
	
	/*metricType:
	 * 	0: CosineDissimilarity
	 * 	1: L1Distance
	 * 	2: EuclideanDistance
	 * 
	 * energyPercentage:
	 *  PCA: components = samples * energyPercentage
	 *  LDA: components = (c-1) *energyPercentage
	 *  LLP: components = (c-1) *energyPercentage
	 * 
	 * featureExtractionMode
	 * 	0: PCA
	 *	1: LDA
	 * 	2: LLP
	 * 
	 * trainNums: how many numbers in 1..10 are assigned to be training faces
	 * for each class, randomly generate the set
	 * 
	 * knn_k: number of K for KNN algorithm
	 * 
	 * */
	static void test(int metricType, double energyPercentage, int featureExtractionMode, int trainNums, int knn_k){
		//determine which metric is used
		//metric
		Metric metric = null;
		if(metricType == 0)
			metric = new CosineDissimilarity();
		else if (metricType == 1)
			metric = new L1Distance();
		else if (metricType == 2)
			metric = new EuclideanDistance();
		
		assert metric != null : "metricType is wrong!";
		
		//set expectedComponents according to energyPercentage
		//componentsRetained
//		int trainingSize = trainNums * 40;
//		int componentsRetained = 0;
//		if(featureExtractionMode == 0)
//			componentsRetained = (int) (trainingSize * energyPercentage);
//		else if(featureExtractionMode == 1)
//			componentsRetained = (int) ((40 -1) * energyPercentage);
//		else if(featureExtractionMode == 2)
//			componentsRetained = (int) ((40 -1) * energyPercentage);
//		
//		assert componentsRetained != 0 : "featureExtractionMode is wrong!";
		
		//set trainSet and testSet
		HashMap<String, ArrayList<Integer>> trainMap = new HashMap();
		HashMap<String, ArrayList<Integer>> testMap = new HashMap();
		for(int i = 1; i <= 40; i ++ ){
			String label = "s"+i;
			ArrayList<Integer> train = generateTrainNums(trainNums);
			ArrayList<Integer> test = generateTestNums(train);
			trainMap.put(label, train);
			testMap.put(label, test);
		}
		
		//trainingSet & respective labels
		ArrayList<Matrix> trainingSet = new ArrayList<Matrix>();
		ArrayList<String> labels = new ArrayList<String>();
		
		Set<String> labelSet = trainMap.keySet();
		Iterator<String> it = labelSet.iterator();
		while(it.hasNext()){
			String label = it.next();
			ArrayList<Integer> cases = trainMap.get(label);
			for(int i = 0; i < cases.size(); i ++){
				String filePath = "faces/"+label+"/"+cases.get(i)+".pgm";
				Matrix temp;
				try {
					temp = FileManager.convertPGMtoMatrix(filePath);
					trainingSet.add(vectorize(temp));
					labels.add(label);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
		}
		
		//testingSet & respective true labels
		ArrayList<Matrix> testingSet = new ArrayList<Matrix>();
		ArrayList<String> trueLabels = new ArrayList<String>();
		
		labelSet = testMap.keySet();
		it = labelSet.iterator();
		while(it.hasNext()){
			String label = it.next();
			ArrayList<Integer> cases = testMap.get(label);
			for(int i = 0; i < cases.size(); i ++){
				String filePath = "faces/"+label+"/"+cases.get(i)+".pgm";
				Matrix temp;
				try {
					temp = FileManager.convertPGMtoMatrix(filePath);
					testingSet.add(vectorize(temp));
					trueLabels.add(label);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
		}
		
		//set featureExtraction
		FeatureExtraction fe = null;
		if(featureExtractionMode == 0)
			fe = new PCA(trainingSet, labels,energyPercentage );
		
		//use test cases to validate
		//testingSet   trueLables
		ArrayList<projectedTrainingMatrix> projectedTrainingSet = fe.getProjectedTrainingSet();
		int accurateNum = 0;
		for(int i = 0 ; i < testingSet.size(); i ++){
			Matrix testCase = fe.getW().transpose().times(testingSet.get(i).minus(fe.getMeanMatrix()));
			String result = KNN.assignLabel(projectedTrainingSet.toArray(new projectedTrainingMatrix[0]), testCase, knn_k, metric);
			//System.out.println(result+"=="+trueLabels.get(i));
			if(result == trueLabels.get(i))
				accurateNum ++;
		}
		System.out.println("The accuracy is "+accurateNum / (double)testingSet.size());
		
	}
	
	
	
	static ArrayList<Integer> generateTrainNums(int trainNum){
		Random random = new Random();
		ArrayList<Integer> result = new ArrayList();
		
		while(result.size() < trainNum){
			int temp = random.nextInt(10)+1;
			while(result.contains(temp)){
				temp = random.nextInt(10)+1;
			}
			result.add(temp);
		}
		
		return result;
	}
	
	static ArrayList<Integer> generateTestNums(ArrayList<Integer> trainSet){
		ArrayList<Integer> result = new ArrayList<Integer>();
		for(int i= 1; i <= 10; i ++){
			if(!trainSet.contains(i))
				result.add(i);
		}
		return result;
	}
	
	// Convert a m by n matrix into a m*n by 1 matrix
	static Matrix vectorize(Matrix input) {
		int m = input.getRowDimension();
		int n = input.getColumnDimension();
		
		Matrix result = new Matrix(m * n, 1);
		for (int p = 0; p < m; p++) {
			for (int q = 0; q < n; q++) {
				result.set(p * n + q, 0, input.get(p, q));
			}
		}
		
		return result;
	}
	
}
