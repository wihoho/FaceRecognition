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

	abstract Matrix getW();

	abstract ArrayList<projectedTrainingMatrix> getProjectedTrainingSet();

	abstract Matrix getMeanMatrix();
}
