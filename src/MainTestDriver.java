import java.io.IOException;
import java.util.ArrayList;

import Jama.Matrix;


public class MainTestDriver {
	public static void main(String[] args){
		try{				       		      	
			Matrix s1 = FileManager.convertPGMtoMatrix("faces\\s1\\1.pgm");
			Matrix s2 = FileManager.convertPGMtoMatrix("faces\\s1\\2.pgm");
			Matrix s3 = FileManager.convertPGMtoMatrix("faces\\s2\\1.pgm");
			Matrix s4 = FileManager.convertPGMtoMatrix("faces\\s2\\2.pgm");
			
			Matrix testS3 = FileManager.convertPGMtoMatrix("faces\\s3\\10.pgm");
			
			ArrayList<Matrix> trainSet = new ArrayList();
			trainSet.add(vectorize(s1));
			trainSet.add(vectorize(s2));
			trainSet.add(vectorize(s3));
			trainSet.add(vectorize(s4));
			
			ArrayList<String> label = new ArrayList<String>();
			label.add("s1");
			label.add("s1");
			label.add("s2");
			label.add("s2");
			
			PCA pca = new PCA(trainSet, label, 2);
			
			Matrix eigen = normalize(pca.getW().getMatrix(0, 10303, 0, 0));
			Matrix testCase = pca.getW().transpose().times(vectorize(testS3).minus(pca.meanMatrix));
			Metric metric = new EuclideanDistance();
			ArrayList<projectedTrainingMatrix> trainingSet =  pca.getProjectedTrainingSet();
			
			String result = KNN.assignLabel(trainingSet.toArray(new projectedTrainingMatrix[0]), testCase, 2, metric);
			System.out.println(result);
		}
		catch(IOException e){
			System.out.println(e.getMessage());
		}
	}
	
	//Convert a m by n matrix into a m*n by 1 matrix
	static Matrix vectorize(Matrix input){
		int m = input.getRowDimension();
		int n = input.getColumnDimension();
		
		Matrix result = new Matrix(m*n,1);
		for(int p = 0; p < n; p ++){
			for(int q = 0; q < m; q ++){
				result.set(p*m+q, 0, input.get(q, p));
			}
		}
		return result;
	}
	
	static Matrix normalize(Matrix input){
		int row = input.getRowDimension();
		
		for(int i = 0; i < row; i ++){
			input.set(i, 0, 0-input.get(i, 0));
			
		}
		
		double max = input.get(0, 0);
		double min = input.get(0, 0);
		
		for(int i = 1; i < row; i ++){
			if(max < input.get(i,0))
				max = input.get(i, 0);
			
			if(min > input.get(i, 0))
				min = input.get(i, 0);
			
		}
		
		Matrix result = new Matrix(112,92);
		for(int p = 0; p < 92; p ++){
			for(int q = 0; q < 112; q ++){
				double value = input.get(p*112+q, 0);
				value = (value - min) *255 /(max - min);
				result.set(q, p, value);
			}
		}
		
		return result;
		
	}
}
