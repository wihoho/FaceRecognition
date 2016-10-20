package com.github.wihoho;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.github.wihoho.constant.FeatureType;
import com.github.wihoho.jama.Matrix;
import com.github.wihoho.training.CosineDissimilarity;
import com.github.wihoho.training.FileManager;

import static org.junit.Assert.assertEquals;

public class TrainerTest {
    ClassLoader classLoader = getClass().getClassLoader();

    @Test
    public void testTraining() throws Exception {
        // Build a trainer
        Trainer trainer = Trainer.builder()
                .metric(new CosineDissimilarity())
                .featureType(FeatureType.PCA)
                .numberOfComponents(3)
                .k(1)
                .build();

        String john1 = "faces/s2/1.pgm";
        String john2 = "faces/s2/2.pgm";
        String john3 = "faces/s2/3.pgm";
        String john4 = "faces/s2/4.pgm";

        String smith1 = "faces/s4/1.pgm";
        String smith2 = "faces/s4/2.pgm";
        String smith3 = "faces/s4/3.pgm";
        String smith4 = "faces/s4/4.pgm";

        // add training data
        trainer.add(convertToMatrix(john1), "john");
        trainer.add(convertToMatrix(john2), "john");
        trainer.add(convertToMatrix(john3), "john");

        trainer.add(convertToMatrix(smith1), "smith");
        trainer.add(convertToMatrix(smith2), "smith");
        trainer.add(convertToMatrix(smith3), "smith");

        // train
        trainer.train();

        // recognize
        assertEquals("john", trainer.recognize(convertToMatrix(john4)));
        assertEquals("smith", trainer.recognize(convertToMatrix(smith4)));
    }

    private Matrix convertToMatrix(String fileAddress) throws IOException {
        File file = new File(classLoader.getResource(fileAddress).getFile());
        return vectorize(FileManager.convertPGMtoMatrix(file.getAbsolutePath()));
    }

    //Convert a m by n matrix into a m*n by 1 matrix
    static Matrix vectorize(Matrix input) {
        int m = input.getRowDimension();
        int n = input.getColumnDimension();

        Matrix result = new Matrix(m * n, 1);
        for (int p = 0; p < n; p++) {
            for (int q = 0; q < m; q++) {
                result.set(p * m + q, 0, input.get(q, p));
            }
        }
        return result;
    }

}