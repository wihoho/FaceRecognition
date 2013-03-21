package Training;
import java.io.IOException;
import java.util.ArrayList;

import Jama.Matrix;

public class MainTestDriver {
	public void  {
		try {
//			Matrix s1 = FileManager.convertPGMtoMatrix("faces/s1/1.pgm");
//			Matrix s2 = FileManager.convertPGMtoMatrix("faces/s1/2.pgm");
//			Matrix s3 = FileManager.convertPGMtoMatrix("faces/s2/1.pgm");
//			Matrix s4 = FileManager.convertPGMtoMatrix("faces/s2/2.pgm");
//
//			Matrix testS3 = FileManager.convertPGMtoMatrix("faces/s2/10.pgm");
//
//			ArrayList<Matrix> trainSet = new ArrayList();
//			trainSet.add(vectorize(s1));
//			trainSet.add(vectorize(s2));
//			trainSet.add(vectorize(s3));
//			trainSet.add(vectorize(s4));
//
//			ArrayList<String> label = new ArrayList<String>();
//			label.add("s3");
//			label.add("s3");
//			label.add("s2");
//			label.add("s2");

			LDA pca = new LDA(trainSet, label, 2);

			Matrix testCase = pca.getW().transpose()
					.times(vectorize(testS3).minus(pca.meanMatrix));
			Metric metric = new EuclideanDistance();
			ArrayList<projectedTrainingMatrix> trainingSet = pca
					.getProjectedTrainingSet();

			String result = KNN.assignLabel(
					trainingSet.toArray(new projectedTrainingMatrix[0]),
					testCase, 2, metric);
			System.out.println(result);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
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
