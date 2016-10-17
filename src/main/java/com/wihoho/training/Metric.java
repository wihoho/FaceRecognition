package com.wihoho.training;

import com.wihoho.jama.Matrix;

public interface Metric {
	double getDistance(Matrix a, Matrix b);
}
