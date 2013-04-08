package Training;
import java.util.ArrayList;
import java.util.Arrays;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class PCA extends FeatureExtraction {

	public PCA(ArrayList<Matrix> trainingSet, ArrayList<String> labels,
			int numOfComponents) throws Exception {
		
		if(numOfComponents >= trainingSet.size()){
			throw new Exception("the expected dimensions could not be achieved!");
		}
		
		this.trainingSet = trainingSet;
		this.labels = labels;
		this.numOfComponents = numOfComponents;

		this.meanMatrix = getMean(this.trainingSet);
		this.W = getFeature(this.trainingSet, this.numOfComponents);

		// Construct projectedTrainingMatrix
		this.projectedTrainingSet = new ArrayList<projectedTrainingMatrix>();
		for (int i = 0; i < trainingSet.size(); i++) {
			projectedTrainingMatrix ptm = new projectedTrainingMatrix(this.W
					.transpose().times(trainingSet.get(i).minus(meanMatrix)),
					labels.get(i));
			this.projectedTrainingSet.add(ptm);
		}
	}

	// extract features, namely W
	private Matrix getFeature(ArrayList<Matrix> input, int K) {
		int i, j;

		int row = input.get(0).getRowDimension();
		int column = input.size();
		Matrix X = new Matrix(row, column);

		for (i = 0; i < column; i++) {
			X.setMatrix(0, row - 1, i, i, input.get(i).minus(this.meanMatrix));
		}

		// get eigenvalues and eigenvectors
		Matrix XT = X.transpose();
		Matrix XTX = XT.times(X);
		EigenvalueDecomposition feature = XTX.eig();
		double[] d = feature.getd();

		assert d.length >= K : "number of eigenvalues is less than K";
		int[] indexes = this.getIndexesOfKEigenvalues(d, K);

		Matrix eigenVectors = X.times(feature.getV());
		Matrix selectedEigenVectors = eigenVectors.getMatrix(0,
				eigenVectors.getRowDimension() - 1, indexes);

		// normalize the eigenvectors
		row = selectedEigenVectors.getRowDimension();
		column = selectedEigenVectors.getColumnDimension();
		for (i = 0; i < column; i++) {
			double temp = 0;
			for (j = 0; j < row; j++)
				temp += Math.pow(selectedEigenVectors.get(j, i), 2);
			temp = Math.sqrt(temp);

			for (j = 0; j < row; j++) {
				selectedEigenVectors.set(j, i, selectedEigenVectors.get(j, i) / temp);
			}
		}

		return selectedEigenVectors;

	}

	// get the first K indexes with the highest eigenValues
	private class mix implements Comparable {
		int index;
		double value;

		mix(int i, double v) {
			index = i;
			value = v;
		}

		@Override
		public int compareTo(Object o) {
			double target = ((mix) o).value;
			if (value > target)
				return -1;
			else if (value < target)
				return 1;

			return 0;
		}
	}

	private int[] getIndexesOfKEigenvalues(double[] d, int k) {
		mix[] mixes = new mix[d.length];
		int i;
		for (i = 0; i < d.length; i++)
			mixes[i] = new mix(i, d[i]);

		Arrays.sort(mixes);

		int[] result = new int[k];
		for (i = 0; i < k; i++)
			result[i] = mixes[i].index;
		return result;
	}

	// The matrix has already been vectorized
	private static Matrix getMean(ArrayList<Matrix> input) {
		int rows = input.get(0).getRowDimension();
		int length = input.size();
		Matrix all = new Matrix(rows, 1);

		for (int i = 0; i < length; i++) {
			all.plusEquals(input.get(i));
		}

		return all.times((double) 1 / length);
	}

	@Override
	public Matrix getW() {
		return this.W;
	}

	@Override
	public ArrayList<projectedTrainingMatrix> getProjectedTrainingSet() {
		return this.projectedTrainingSet;
	}
	
	@Override
	public Matrix getMeanMatrix() {
		// TODO Auto-generated method stub
		return meanMatrix;
	}
	
	public ArrayList<Matrix> getTrainingSet(){
		return this.trainingSet;
	}
	
	public Matrix reconstruct(int whichImage, int dimensions) throws Exception{
		if(dimensions > this.numOfComponents)
			throw new Exception("dimensions should be smaller than the number of components");
		
		Matrix afterPCA = this.projectedTrainingSet.get(whichImage).matrix.getMatrix(0, dimensions-1, 0, 0);
		Matrix eigen = this.getW().getMatrix(0, 10304-1, 0, dimensions - 1);
		return eigen.times(afterPCA).plus(this.getMeanMatrix());
		
	}

}
