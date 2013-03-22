package Validation;

import Training.EuclideanDistance;
import Training.FeatureExtraction;
import Training.FileManager;
import Training.KNN;
import Training.Metric;
import Training.PCA;
import Training.LDA;
import Training.LLP;
import Training.projectedTrainingMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class ValidationMain {
	
	public static final int PCA = 1;
	public static final int LDA = 2;
	public static final int LLP = 3;
	
	public static final int RANDOM_SAMPLING = 1;
	public static final int K_FOLD = 2;
	public static final int LEAVE_ONE_OUT = 3;
	public static final int XU_DONG = 4;
	
	public static int k = 5;
	public static int random_samples = 20;
	public static int xu_dong_random_samples = 2;
	
	public static int section = 10;
	public static int folder = 10;
	
	public static void main(String[] args) throws IOException {
		
		int type = LDA;
		int validation = XU_DONG;
		int numOfComponents = 2;
		int knn_k = 2;
		
//		int section = 10;
//		int folder = 10;
		
		ArrayList<Matrix> trainSet = new ArrayList();
		ArrayList<String> label = new ArrayList<String>();

//		for (int i = 1; i <= section; i++){
//			for (int j = 1; j <= folder; j++){
//				String file = "faces/s" + String.valueOf(j) + "/" + String.valueOf(i) + ".pgm";
//				trainSet.add(vectorize(FileManager.convertPGMtoMatrix(file)));
//				label.add("s" + String.valueOf(j));
//				System.out.println("add " + "faces/s" + String.valueOf(j) + "/" + String.valueOf(i) + ".pgm");
//				System.out.println("Label is s" + String.valueOf(j));
//			}
//		}
		
		for (int i = 1; i <= folder; i++){
			for (int j = 1; j <= section; j++){
				String file = "faces/s" + String.valueOf(i) + "/" + String.valueOf(j) + ".pgm";
				trainSet.add(vectorize(FileManager.convertPGMtoMatrix(file)));
				label.add("s" + String.valueOf(i));
				System.out.println("add " + "faces/s" + String.valueOf(i) + "/" + String.valueOf(j) + ".pgm");
				System.out.println("Label is s" + String.valueOf(i));
			}
		}
		
//		Matrix s1 = FileManager.convertPGMtoMatrix("faces/s1/1.pgm");
//		Matrix s2 = FileManager.convertPGMtoMatrix("faces/s1/2.pgm");
//		Matrix s3 = FileManager.convertPGMtoMatrix("faces/s2/1.pgm");
//		Matrix s4 = FileManager.convertPGMtoMatrix("faces/s2/2.pgm");
//		Matrix s5 = FileManager.convertPGMtoMatrix("faces/s2/10.pgm");
//		
////		Matrix testS3 = FileManager.convertPGMtoMatrix("faces/s2/10.pgm");
//		
//		trainSet.add(vectorize(s1));
//		trainSet.add(vectorize(s2));
//		trainSet.add(vectorize(s3));
//		trainSet.add(vectorize(s4));
//		trainSet.add(vectorize(s5));
//		
//		label.add("s1");
//		label.add("s1");
//		label.add("s2");
//		label.add("s2");
//		label.add("s2");
		
		
		validate(type, trainSet, validation, label, numOfComponents, knn_k);
	}
	
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
	
	static void validate(int type, ArrayList<Matrix> dataset, int validate, ArrayList<String> label, int numOfComponents, int knn_k){
		switch(validate){
		case RANDOM_SAMPLING:
			random_sampling(type, dataset, label, numOfComponents, knn_k);
			break;
		case K_FOLD:
			k_fold(type, dataset, label, numOfComponents, knn_k);
			break;
		case LEAVE_ONE_OUT:
			leave_one_out(type, dataset, label, numOfComponents, knn_k);
			break;
		case XU_DONG:
			xu_dong(type, dataset, label, numOfComponents, knn_k);
		}
		return;
	}

	private static void xu_dong(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels, int numOfComponents, int knn_k) {
		// TODO Auto-generated method stub
		int size = dataset.size();
		double totalAccuracy = 0.0;
		ArrayList<ArrayList<Matrix>> testsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<Matrix>> trainingsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<String>> labelsets = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<String>> testlabels = new ArrayList<ArrayList<String>>();
		
		for (int j = 0; j < k; j++){
			
			ArrayList<Matrix> testingset = new ArrayList<Matrix>();
			ArrayList<Matrix> trainingset = new ArrayList<Matrix>();
			ArrayList<String> labelset = new ArrayList<String>();
			ArrayList<String> testlabel = new ArrayList<String>();
			
			
			ArrayList<Integer> splits = new ArrayList<Integer>();
			for (int p = 0; p < folder; p ++){
				int low = p * section;
				for (int q = 0; q < xu_dong_random_samples; q++){
					int index = (int)(Math.random()*section) + low;
					while (splits.contains(index)){
						index = (int)(Math.random()*section) + low;
					}
					splits.add(index);			
				}
			}
			
			System.out.println("test cases are: ");
			for (int i : splits){
				System.out.print(i + " ");
				testingset.add(dataset.get(i));
				testlabel.add(labels.get(i));
			}
			System.out.println();
			for (int i = 0; i < size; i++){
				if (!splits.contains(i)){
					trainingset.add(dataset.get(i));
					labelset.add(labels.get(i));
				}
			}
			System.out.println("Traning set size : " + trainingset.size());
			
			trainingsets.add(trainingset);
			labelsets.add(labelset);
			testsets.add(testingset);
			testlabels.add(testlabel);
		}
		for (int n = 0; n < k; n++){
			System.out.println("Fold " + n);
			ArrayList<Matrix> train = trainingsets.get(n);
			ArrayList<String> label = labelsets.get(n);
			ArrayList<Matrix> test = testsets.get(n);
			ArrayList<String> testlabel = testlabels.get(n);
			ArrayList<projectedTrainingMatrix> trainingSet;
			Matrix testCase;
			Metric metric;
			String result = "";
			int calc = 0;
			double fold_accuracy = 0;
			switch (type){
			case PCA:
				calc = 0;
				for (int j = 0; j < xu_dong_random_samples*folder; j++){
					PCA pca = new PCA(train, label, numOfComponents);
					testCase = pca.getW().transpose()
							.times(vectorize(test.get(j)).minus(pca.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = pca
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)(xu_dong_random_samples*folder);
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
				
			case LDA:
				calc = 0;
				for (int j = 0; j < (xu_dong_random_samples*folder); j++){
					LDA lda = new LDA(train, label, numOfComponents);
					testCase = lda.getW().transpose()
							.times(vectorize(test.get(j)).minus(lda.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = lda
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)(xu_dong_random_samples*folder);
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
				
			case LLP:
				calc = 0;
				for (int j = 0; j < (xu_dong_random_samples*folder); j++){
					LLP llp = new LLP(train, label, numOfComponents);
					testCase = llp.getW().transpose()
							.times(vectorize(test.get(j)).minus(llp.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = llp
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)(xu_dong_random_samples*folder);
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
			}
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}

	private static void leave_one_out(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels, int numOfComponents, int knn_k) {
		// TODO Auto-generated method stub
		int size = dataset.size();
		ArrayList<Matrix> testsets = new ArrayList<Matrix>();
		ArrayList<ArrayList<Matrix>> trainingsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<String>> labelsets = new ArrayList<ArrayList<String>>();
		for (int i = 0; i < size; i ++){
			System.out.println("Test Case is " + labels.get(i));
			Matrix testCase = dataset.get(i);
			ArrayList<Matrix> trainingset = new ArrayList<Matrix>();
			ArrayList<String> labelset = new ArrayList<String>();
			for (int q = 0; q < i; q++){
				trainingset.add(dataset.get(q));
				labelset.add(labels.get(q));
			}
			for (int p = i+1; p < size; p++){
				trainingset.add(dataset.get(p));
				labelset.add(labels.get(p));
			}
			
			trainingsets.add(trainingset);
			labelsets.add(labelset);
			testsets.add(testCase);
		}
		
		int calc = 0;
		
		for (int j = 0; j < testsets.size(); j++){
			System.out.println("Fold " + j);
			ArrayList<Matrix> train = trainingsets.get(j);
			ArrayList<String> label = labelsets.get(j);
			Matrix test = testsets.get(j);
			Matrix testCase;
			ArrayList<projectedTrainingMatrix> trainingSet;
			String result = "";
			Metric metric;
			switch (type){
			case PCA:
				PCA pca = new PCA(train, label, numOfComponents);
				testCase = pca.getW().transpose()
						.times(vectorize(test).minus(pca.getMeanMatrix()));
				metric = new EuclideanDistance();
				trainingSet = pca
						.getProjectedTrainingSet();

				result = KNN.assignLabel(
						trainingSet.toArray(new projectedTrainingMatrix[0]),
						testCase, knn_k, metric);
//				System.out.println(result);
				break;
			case LDA:
				LDA lda = new LDA(train, label, numOfComponents);
				testCase = lda.getW().transpose()
						.times(vectorize(test).minus(lda.getMeanMatrix()));
				metric = new EuclideanDistance();
				trainingSet = lda
						.getProjectedTrainingSet();

				result = KNN.assignLabel(
						trainingSet.toArray(new projectedTrainingMatrix[0]),
						testCase, knn_k, metric);
//				System.out.println(result);
				break;
			case LLP:
				LLP llp = new LLP(train, label, numOfComponents);
				testCase = llp.getW().transpose()
						.times(vectorize(test).minus(llp.getMeanMatrix()));
				metric = new EuclideanDistance();
				trainingSet = llp
						.getProjectedTrainingSet();

				result = KNN.assignLabel(
						trainingSet.toArray(new projectedTrainingMatrix[0]),
						testCase, knn_k, metric);
//				System.out.println(result);
				break;
				
			}
//			System.out.println("Result is : " + result);
//			System.out.println("Label is : " + labels.get(j));
//			
			if (result.equals(labels.get(j))){
				calc ++;
			}
//			System.out.println();
			
		}
		System.out.println("Correct predictions : " + calc);
		System.out.println("Total test cases : " + dataset.size());
		System.out.println("testing accuracy: " + ((double)calc/(double)dataset.size()) * 100 + "%");
		
	}

	private static void k_fold(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels, int numOfComponents, int knn_k) {
		// TODO Auto-generated method stub
		int size = dataset.size();
		int fold_size = size/k;
		ArrayList<ArrayList<Matrix>> testsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<Matrix>> trainingsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<String>> labelsets = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<String>> testlabels = new ArrayList<ArrayList<String>>();
		for (int i = 0; i < k; i ++){
			int low = i * fold_size;
			int high = low + fold_size;
//			System.out.println("low is " + low);
//			System.out.println("High is " + high);
			ArrayList<Matrix> testingset = new ArrayList<Matrix>();
			ArrayList<Matrix> trainingset = new ArrayList<Matrix>();
			ArrayList<String> labelset = new ArrayList<String>();
			ArrayList<String> testlabel = new ArrayList<String>();
			for (int m = 0; m < low; m++){
				trainingset.add(dataset.get(m));
				labelset.add(labels.get(m));
			}
			for (int m = high; m < size; m++){
				trainingset.add(dataset.get(m));
				labelset.add(labels.get(m));
			}
			for (int m = low; m < high; m++){
				testingset.add(dataset.get(m));
				testlabel.add(labels.get(m));
			}
			
			trainingsets.add(trainingset);
			labelsets.add(labelset);
			testsets.add(testingset);
			testlabels.add(testlabel);
		}
		double totalAccuracy = 0.0;
		for (int i = 0; i < k; i++){
			System.out.println("Fold " + i);
			ArrayList<Matrix> train = trainingsets.get(i);
			ArrayList<String> label = labelsets.get(i);
			ArrayList<Matrix> test = testsets.get(i);
			ArrayList<String> testlabel = testlabels.get(i);
			ArrayList<projectedTrainingMatrix> trainingSet;
			Matrix testCase;
			Metric metric;
			String result = "";
			int calc = 0;
			double fold_accuracy = 0;
			switch(type){
			case PCA:
				calc = 0;
				for (int j = 0; j < fold_size; j++){
					PCA pca = new PCA(train, label, numOfComponents);
					testCase = pca.getW().transpose()
							.times(vectorize(test.get(j)).minus(pca.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = pca
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)fold_size;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
			case LDA:
				calc = 0;
				for (int j = 0; j < fold_size; j++){
					LDA lda = new LDA(train, label, numOfComponents);
					testCase = lda.getW().transpose()
							.times(vectorize(test.get(j)).minus(lda.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = lda
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)fold_size;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
			case LLP:
				calc = 0;
				for (int j = 0; j < fold_size; j++){
					LLP pca = new LLP(train, label, numOfComponents);
					testCase = pca.getW().transpose()
							.times(vectorize(test.get(j)).minus(pca.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = pca
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)fold_size;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
			}
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}

	private static void random_sampling(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels, int numOfComponents, int knn_k) {
		// TODO Auto-generated method stub
		int size = dataset.size();
		double totalAccuracy = 0.0;
		ArrayList<ArrayList<Matrix>> testsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<Matrix>> trainingsets = new ArrayList<ArrayList<Matrix>>();
		ArrayList<ArrayList<String>> labelsets = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<String>> testlabels = new ArrayList<ArrayList<String>>();

		for (int j = 0; j < k; j++){
			ArrayList<Integer> splits = new ArrayList<Integer>();
			ArrayList<Matrix> testingset = new ArrayList<Matrix>();
			ArrayList<Matrix> trainingset = new ArrayList<Matrix>();
			ArrayList<String> labelset = new ArrayList<String>();
			ArrayList<String> testlabel = new ArrayList<String>();
			
			for (int i = 0; i < random_samples; i++){
				int index = (int)(Math.random()*size);
				while (splits.contains(index)){
					index = (int)(Math.random()*size);
				}
				splits.add(index);			
			}
			System.out.println("test cases are: ");
			for (int i : splits){
				System.out.print(i + " ");
				testingset.add(dataset.get(i));
				testlabel.add(labels.get(i));
			}
			System.out.println();
			for (int i = 0; i < size; i++){
				if (!splits.contains(i)){
					trainingset.add(dataset.get(i));
					labelset.add(labels.get(i));
				}
			}
			System.out.println("Traning set size : " + trainingset.size());
			
			trainingsets.add(trainingset);
			labelsets.add(labelset);
			testsets.add(testingset);
			testlabels.add(testlabel);
		}
		for (int n = 0; n < k; n++){
			System.out.println("Fold " + n);
			ArrayList<Matrix> train = trainingsets.get(n);
			ArrayList<String> label = labelsets.get(n);
			ArrayList<Matrix> test = testsets.get(n);
			ArrayList<String> testlabel = testlabels.get(n);
			ArrayList<projectedTrainingMatrix> trainingSet;
			Matrix testCase;
			Metric metric;
			String result = "";
			int calc = 0;
			double fold_accuracy = 0;
			switch (type){
			case PCA:
				calc = 0;
				for (int j = 0; j < random_samples; j++){
					PCA pca = new PCA(train, label, numOfComponents);
					testCase = pca.getW().transpose()
							.times(vectorize(test.get(j)).minus(pca.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = pca
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)random_samples;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
				
			case LDA:
				calc = 0;
				for (int j = 0; j < random_samples; j++){
					LDA lda = new LDA(train, label, numOfComponents);
					testCase = lda.getW().transpose()
							.times(vectorize(test.get(j)).minus(lda.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = lda
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)random_samples;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
				
			case LLP:
				calc = 0;
				for (int j = 0; j < random_samples; j++){
					LLP llp = new LLP(train, label, numOfComponents);
					testCase = llp.getW().transpose()
							.times(vectorize(test.get(j)).minus(llp.getMeanMatrix()));
					metric = new EuclideanDistance();
					trainingSet = llp
							.getProjectedTrainingSet();

					result = KNN.assignLabel(
							trainingSet.toArray(new projectedTrainingMatrix[0]),
							testCase, knn_k, metric);
					System.out.print(result + " == " + testlabel.get(j) + " ");
					if (result.equals(testlabel.get(j))){
						calc++;
					}
				}
				fold_accuracy = (double)calc/(double)random_samples;
				System.out.println(fold_accuracy);
				totalAccuracy += fold_accuracy;
				break;
			}
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}
	
}
