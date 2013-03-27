package Training;
import Jama.Matrix;

public interface Metric {
	double getDistance(Matrix a, Matrix b);
}
