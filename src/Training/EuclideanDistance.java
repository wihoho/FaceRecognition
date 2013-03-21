package Training;
import Jama.Matrix;

public class EuclideanDistance implements Metric {

	@Override
	public double getDistance(Matrix a, Matrix b) {
		int size = a.getRowDimension();
		double sum = 0;

		for (int i = 0; i < size; i++) {
			sum += Math.pow(a.get(i, 0) - b.get(i, 0), 2);
		}

		return Math.sqrt(sum);
	}

}
