package com.github.wihoho.training;

import com.github.wihoho.jama.Matrix;

public interface Metric {
	double getDistance(Matrix a, Matrix b);
}
