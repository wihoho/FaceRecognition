package Training;
import Jama.Matrix;

public class projectedTrainingMatrix {
	Matrix matrix;
	String label;
	double distance = 0;

	public projectedTrainingMatrix(Matrix m, String l) {
		this.matrix = m;
		this.label = l;
	}
}
