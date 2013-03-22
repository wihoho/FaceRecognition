package Validation;

import Training.CosineDissimilarity;
import Training.EuclideanDistance;
import Training.FeatureExtraction;
import Training.FileManager;
import Training.KNN;
import Training.L1Distance;
import Training.Metric;
import Training.PCA;
import Training.LDA;
import Training.LLP;
import Training.projectedTrainingMatrix;

import java.io.IOException;
import java.io.ObjectInputStream.GetField;
import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class Validation {
	
	public static final int PCA = 1;
	public static final int LDA = 2;
	public static final int LLP = 3;
	
	public static final int RANDOM_SAMPLING = 1;
	public static final int K_FOLD = 2;
	public static final int LEAVE_ONE_OUT = 3;
	public static final int XU_DONG = 4;
	
	public static final int CosineDisimilarity = 0;
	public static final int L1Distance = 1;
	public static final int Euclidean = 2;
	
	public static int k = 5;
	public static int random_samples = 20;
	public static int xu_dong_random_samples = 2;
	
	public static int metric_type = L1Distance;
	
	public static int section = 10;
	public static int folder = 10;
	public static Metric metric;
	public static double energy = 0.5;
	public static int type = LDA;
	public static int validation = XU_DONG;
	public static int knn_k = 2;
	private static ArrayList<Matrix> trainSet = new ArrayList<Matrix>();
	private static ArrayList<String> label = new ArrayList<String>();
	
	public Validation(int xu_dong_random_samples, int folds, int metric_type, double energy, int type, int validation, int knn_k) throws IOException{
		this.xu_dong_random_samples  = xu_dong_random_samples;
		this.k = folds;
		this.metric_type = metric_type;
		this.energy = energy;
		this.type = type;
		this.validation = validation;
		this.knn_k = knn_k;
		switch(metric_type){
		case CosineDisimilarity:
			metric = new CosineDissimilarity();
			break;
		case L1Distance:
			metric = new L1Distance();
			break;
		case Euclidean:
			metric = new EuclideanDistance();				
		}
		initialize();
	}
	
	public void initialize() throws IOException {
		
		for (int i = 1; i <= folder; i++){
			for (int j = 1; j <= section; j++){
				String file = "faces/s" + String.valueOf(i) + "/" + String.valueOf(j) + ".pgm";
				trainSet.add(vectorize(FileManager.convertPGMtoMatrix(file)));
				label.add("s" + String.valueOf(i));
				System.out.println("add " + "faces/s" + String.valueOf(i) + "/" + String.valueOf(j) + ".pgm");
				System.out.println("Label is s" + String.valueOf(i));
			}
		}		
//		validate(type, trainSet, validation, label);
	}
	
	public void validate(){
		validate(type, trainSet, validation, label);
	}
	
	private Matrix vectorize(Matrix input) {
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
	
	public void setVariables(int xu_dong_random_samples, int folds, int metric_type, double energy, int type, int validation, int knn_k){
		this.xu_dong_random_samples  = xu_dong_random_samples;
		this.k = folds;
		this.metric_type = metric_type;
		this.energy = energy;
		this.type = type;
		this.validation = validation;
		this.knn_k = knn_k;
		switch(metric_type){
		case CosineDisimilarity:
			metric = new CosineDissimilarity();
			break;
		case L1Distance:
			metric = new L1Distance();
			break;
		case Euclidean:
			metric = new EuclideanDistance();				
		}
	}
	
	private int findNumOfComponents(int training_size){
		return (int)(training_size*energy);
	}
	
	public void validate(int type, ArrayList<Matrix> dataset, int validate, ArrayList<String> label){
		switch(validate){
		case RANDOM_SAMPLING:
			random_sampling(type, dataset, label);
			break;
		case K_FOLD:
			k_fold(type, dataset, label);
			break;
		case LEAVE_ONE_OUT:
			leave_one_out(type, dataset, label);
			break;
		case XU_DONG:
			xu_dong(type, dataset, label);
		}
		return;
	}

	private void xu_dong(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels) {
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
			String result = "";
			int calc = 0;
			double fold_accuracy = 0;
			FeatureExtraction fe = null;
			switch (type){
			case PCA:
				fe = new PCA(train, labels, findNumOfComponents(train.size()));
				break;
			case LDA:
				fe = new LDA(train, labels, findNumOfComponents(train.size()));
				break;
			case LLP: 
				fe = new LLP(train, labels, findNumOfComponents(train.size()));
				break;
			}
			calc = 0;
			for (int j = 0; j < xu_dong_random_samples*folder; j++){
				testCase = fe.getW().transpose()
						.times(vectorize(test.get(j)).minus(fe.getMeanMatrix()));
				trainingSet = fe
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
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}

	private void leave_one_out(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels) {
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
			FeatureExtraction fe = null;
			switch (type){
			case PCA:
				fe = new PCA(train, labels, findNumOfComponents(train.size()));
				break;
			case LDA:
				fe = new LDA(train, labels, findNumOfComponents(train.size()));
				break;
			case LLP:
				fe = new LLP(train, labels, findNumOfComponents(train.size()));
				break;
			}
			testCase = fe.getW().transpose()
					.times(vectorize(test).minus(fe.getMeanMatrix()));
			trainingSet = fe
					.getProjectedTrainingSet();

			result = KNN.assignLabel(
					trainingSet.toArray(new projectedTrainingMatrix[0]),
					testCase, knn_k, metric);
				
			if (result.equals(labels.get(j))){
				calc ++;
			}
//			System.out.println();
			
		}
		System.out.println("Correct predictions : " + calc);
		System.out.println("Total test cases : " + dataset.size());
		System.out.println("testing accuracy: " + ((double)calc/(double)dataset.size()) * 100 + "%");
		
	}

	private void k_fold(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels) {
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
			String result = "";
			int calc = 0;
			double fold_accuracy = 0;
			FeatureExtraction fe = null;
			switch(type){
			case PCA:
				fe = new PCA(train, labels, findNumOfComponents(train.size()));
				break;
			case LDA:
				fe = new LDA(train, labels, findNumOfComponents(train.size()));
				break;
			case LLP:
				fe = new LLP(train, labels, findNumOfComponents(train.size()));
				break;
			}
			calc = 0;
			for (int j = 0; j < fold_size; j++){
				testCase = fe.getW().transpose()
						.times(vectorize(test.get(j)).minus(fe.getMeanMatrix()));
				trainingSet = fe
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
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}

	private void random_sampling(int type, ArrayList<Matrix> dataset,
			ArrayList<String> labels) {
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
			String result = "";
			FeatureExtraction fe = null;
			int calc = 0;
			double fold_accuracy = 0;
			switch (type){
			case PCA:
				fe = new PCA(train, labels, findNumOfComponents(train.size()));
				break;
			case LDA:
				fe = new LDA(train, labels, findNumOfComponents(train.size()));
				break;
			case LLP:
				fe = new LLP(train, labels, findNumOfComponents(train.size()));
				break;
			}
			calc = 0;
			for (int j = 0; j < random_samples; j++){
				testCase = fe.getW().transpose()
						.times(vectorize(test.get(j)).minus(fe.getMeanMatrix()));
				trainingSet = fe
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
		}
		System.out.println("Average accuracy is : " + (totalAccuracy/(double)k * 100) + "%");
		
	}
	
}
