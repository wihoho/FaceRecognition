package com.wihoho.training;
import java.util.ArrayList;

import com.wihoho.jama.Matrix;


public abstract class FeatureExtraction {
	ArrayList<Matrix> trainingSet;
	ArrayList<String> labels;
	int numOfComponents;
	Matrix meanMatrix;
	// Output
	Matrix W;
	ArrayList<ProjectedTrainingMatrix> projectedTrainingSet;

	public abstract Matrix getW();

	public abstract ArrayList<ProjectedTrainingMatrix> getProjectedTrainingSet();

	public abstract Matrix getMeanMatrix();
}
