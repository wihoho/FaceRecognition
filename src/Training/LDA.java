package Training;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class LDA extends FeatureExtraction {

	public LDA(ArrayList<Matrix> trainingSet, ArrayList<String> labels,
		int numOfComponents) throws Exception {
		int n = trainingSet.size(); // sample size
		Set<String> tempSet = new HashSet<String>(labels);
		int c = tempSet.size(); // class size
		assert numOfComponents >= n - c : "the input components is smaller than n - c!";
		assert n >= 2 * c : "n is smaller than 2c!";

		// process in PCA
		PCA pca = new PCA(trainingSet, labels, n - c);

		// classify
		Matrix meanTotal = new Matrix(n - c, 1);

		HashMap<String, ArrayList<Matrix>> map = new HashMap<String, ArrayList<Matrix>>();
		ArrayList<projectedTrainingMatrix> pcaTrain = pca
				.getProjectedTrainingSet();
		for (int i = 0; i < pcaTrain.size(); i++) {
			String key = pcaTrain.get(i).label;
			meanTotal.plusEquals(pcaTrain.get(i).matrix);
			if (!map.containsKey(key)) {
				ArrayList<Matrix> temp = new ArrayList<Matrix>();
				temp.add(pcaTrain.get(i).matrix);
				map.put(key, temp);
			} else {
				ArrayList<Matrix> temp = map.get(key);
				temp.add(pcaTrain.get(i).matrix);
				map.put(key, temp);
			}
		}
		meanTotal.times((double) 1 / n);

		// calculate Sw, Sb
		Matrix Sw = new Matrix(n - c, n - c);
		Matrix Sb = new Matrix(n - c, n - c);

		tempSet = map.keySet();
		Iterator<String> it = tempSet.iterator();
		while (it.hasNext()) {
			String s = (String) it.next();
			ArrayList<Matrix> matrixWithinThatClass = map.get(s);
			Matrix meanOfCurrentClass = getMean(matrixWithinThatClass);
			for (int i = 0; i < matrixWithinThatClass.size(); i++) {
				Matrix temp1 = matrixWithinThatClass.get(i).minus(
						meanOfCurrentClass);
				temp1 = temp1.times(temp1.transpose());
				Sw.plusEquals(temp1);
			}

			Matrix temp = meanOfCurrentClass.minus(meanTotal);
			temp = temp.times(temp.transpose()).times(
					matrixWithinThatClass.size());
			Sb.plusEquals(temp);
		}

		// calculate the eigenvalues and vectors of Sw^-1 * Sb
		Matrix targetForEigen = Sw.inverse().times(Sb);
		EigenvalueDecomposition feature = targetForEigen.eig();

		double[] d = feature.getd();
		assert d.length >= c - 1 : "Ensure that the number of eigenvalues is larger than c - 1";
		int[] indexes = getIndexesOfKEigenvalues(d, c - 1);

		Matrix eigenVectors = feature.getV();
		Matrix selectedEigenVectors = eigenVectors.getMatrix(0,
				eigenVectors.getRowDimension() - 1, indexes);

		this.W = pca.getW().times(selectedEigenVectors);

		// Construct projectedTrainingMatrix
		this.projectedTrainingSet = new ArrayList<projectedTrainingMatrix>();
		for (int i = 0; i < trainingSet.size(); i++) {
			projectedTrainingMatrix ptm = new projectedTrainingMatrix(this.W
					.transpose()
					.times(trainingSet.get(i).minus(pca.meanMatrix)),
					labels.get(i));
			this.projectedTrainingSet.add(ptm);
		}
		this.meanMatrix = pca.meanMatrix;
	}

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

	static Matrix getMean(ArrayList<Matrix> m) {
		int num = m.size();
		int row = m.get(0).getRowDimension();
		int column = m.get(0).getColumnDimension();

		assert column == 1 : "expected column does not equal to 1!";

		Matrix mean = new Matrix(row, column);
		for (int i = 0; i < num; i++) {
			mean.plusEquals(m.get(i));
		}

		mean = mean.times((double) 1 / num);
		return mean;
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
}
