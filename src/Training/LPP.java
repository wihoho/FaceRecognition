package Training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;



public class LPP extends FeatureExtraction {

	public LPP(ArrayList<Matrix> trainingSet, ArrayList<String> labels, int numOfComponents) throws Exception{
		int n = trainingSet.size(); // sample size
		Set<String> tempSet = new HashSet<String>(labels);
		int c = tempSet.size(); // class size

		// process in PCA
		PCA pca = new PCA(trainingSet,labels, numOfComponents);

		//construct the nearest neighbor graph 
		Matrix S = constructNearestNeighborGraph(pca.projectedTrainingSet);
		Matrix D = constructD(S);
		Matrix L = D.minus(S);

		//reconstruct the trainingSet into required X;
		Matrix X = constructTrainingMatrix(pca.getProjectedTrainingSet());
		Matrix XLXT = X.times(L).times(X.transpose());
		Matrix XDXT = X.times(D).times(X.transpose());

		//calculate the eignevalues and eigenvectors of (XDXT)^-1 * (XLXT)
		Matrix targetForEigen = XDXT.inverse().times(XLXT);
		EigenvalueDecomposition feature = targetForEigen.eig();

		double[] d = feature.getd();
		assert d.length >= c - 1 :"Ensure that the number of eigenvalues is larger than c - 1";
		int[] indexes = getIndexesOfKEigenvalues(d,d.length);

		Matrix eigenVectors = feature.getV();
		Matrix selectedEigenVectors = eigenVectors.getMatrix(0, eigenVectors.getRowDimension()-1,indexes);

		this.W = pca.getW().times(selectedEigenVectors);

		//Construct projectedTrainingMatrix
		this.projectedTrainingSet = new ArrayList<projectedTrainingMatrix>();
		for(int i = 0; i < trainingSet.size(); i ++){
			projectedTrainingMatrix ptm = new projectedTrainingMatrix(this.W.transpose().times(trainingSet.get(i).minus(pca.meanMatrix)),labels.get(i));
			this.projectedTrainingSet.add(ptm);
		}
		this.meanMatrix = pca.meanMatrix;
	}

	private Matrix constructNearestNeighborGraph(ArrayList<projectedTrainingMatrix> input){
		int size = input.size();
		Matrix S = new Matrix(size, size);
		
		Metric Euclidean = new EuclideanDistance();
		projectedTrainingMatrix[] trainArray = input.toArray(new projectedTrainingMatrix[input.size()]);
		
		for(int i = 0; i < size; i ++){
			projectedTrainingMatrix[] neighbors = KNN.findKNN(trainArray, input.get(i).matrix, 3, Euclidean);
			for(int j = 0; j < neighbors.length; j ++){
				if(!neighbors[j].equals(input.get(i))){
//					double distance = Euclidean.getDistance(neighbors[j].matrix, input.get(i).matrix);
//					double weight = Math.exp(0-distance*distance / 2);
					int index = input.indexOf(neighbors[j]);
					S.set(i, index, 1);
					S.set(index, i,1);
				}
			}
			
//			for(int j = 0; j < size; j ++){
//				if( i != j && input.get(i).label.equals(input.get(j).label)){
//					S.set(i, j, 1);
//				}
//			}
		}
		return S;
	}

	private Matrix constructD(Matrix S){
		int size = S.getRowDimension();
		Matrix D = new Matrix(size, size);

		for(int i = 0; i < size; i++){
			double temp = 0;
			for(int j = 0; j < size; j ++){
				temp += S.get(j, i);
			}
			D.set(i, i, temp);
		}

		return D;
	}

	private Matrix constructTrainingMatrix(ArrayList<projectedTrainingMatrix> input){
		int row = input.get(0).matrix.getRowDimension();
		int column = input.size();
		Matrix X = new Matrix(row, column);

		for(int i = 0; i < column; i ++){
			X.setMatrix(0, row-1, i, i, input.get(i).matrix);
		}

		return X;
	}

	private int[] getIndexesOfKEigenvalues(double[] d, int k){
		mix[] mixes = new mix[d.length];
		int i;
		for(i = 0; i < d.length; i ++)
			mixes[i] = new mix(i, d[i]);

		Arrays.sort(mixes);

		int[] result = new int[k];
		for(i = 0; i < k; i ++)
			result[i] = mixes[i].index;
		return result;
	}

	private class mix implements Comparable{
		int index;
		double value;

		mix(int i, double v){
			index = i;
			value = v;
		}

		@Override
		public int compareTo(Object o) {
			double target = ((mix)o).value;
			if(value > target)
				return -1;
			else if(value < target)
				return 1;

			return 0;
		}
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
		return this.meanMatrix;
	}

}