package com.github.wihoho.training;

import com.github.wihoho.jama.Matrix;

public class L1Distance implements Metric {

	@Override
	public double getDistance(Matrix a, Matrix b) {
		int size = a.getRowDimension();
		double sum = 0;

		for (int i = 0; i < size; i++) {
			sum += Math.abs(a.get(i, 0) - b.get(i, 0));
		}

		return sum;
	}

}
