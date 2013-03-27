package Training;
import java.util.ArrayList;

import Jama.Matrix;

public abstract class FeatureExtraction {
	ArrayList<Matrix> trainingSet;
	ArrayList<String> labels;
	int numOfComponents;
	Matrix meanMatrix;
	// Output
	Matrix W;
	ArrayList<projectedTrainingMatrix> projectedTrainingSet;

	public abstract Matrix getW();

	public abstract ArrayList<projectedTrainingMatrix> getProjectedTrainingSet();

	public abstract Matrix getMeanMatrix();
}
