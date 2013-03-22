package Validation;

import java.io.IOException;
import java.util.Scanner;

import Training.Metric;

public class ValidateMain {
	public static final int PCA = 1;
	public static final int LDA = 2;
	public static final int LLP = 3;
	
	public static final int RANDOM_SAMPLING = 1;
	public static final int K_FOLD = 2;
	public static final int LEAVE_ONE_OUT = 3;
	public static final int XU_DONG = 4;
	
	public static final int CosineDisimilarity = 0;
	public static final int L1Distance = 1;
	public static final int Euclidean = 2;
	
	public static int k = 5;
	public static int random_samples = 20;
	public static int xu_dong_random_samples = 2;
	
	public static int metric_type = L1Distance;
	
	public static int section = 10;
	public static int folder = 10;
	public static Metric metric;
	public static double energy = 0.5;
	public static int type = LDA;
	public static int validation = XU_DONG;
	public static int knn_k = 2;
	
	public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		Validation v = new Validation(0, 0, 0, 0, 0, 0, 0);
		for (int i = 0; i < 2; i++){
			String line = sc.nextLine();
			System.out.println(line);
			String[] params = line.split("\\s");
			v.setVariables(Integer.parseInt(params[0]), Integer.parseInt(params[1]), Integer.parseInt(params[2]), Double.parseDouble(params[3]), Integer.parseInt(params[4]), Integer.parseInt(params[5]), Integer.parseInt(params[6]));
			v.validate();
		}
	}
	
}
