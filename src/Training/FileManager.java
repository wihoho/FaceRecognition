package Training;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

import Jama.Matrix;

public class FileManager {
	// Convert PGM to Matrix
	public static Matrix convertPGMtoMatrix(String address) throws IOException {
		FileInputStream fileInputStream = new FileInputStream(address);
		Scanner scan = new Scanner(fileInputStream);

		// Discard the magic number
		scan.nextLine();
		// Read pic width, height and max value
		int picWidth = scan.nextInt();
		int picHeight = scan.nextInt();
		int maxvalue = scan.nextInt();

		fileInputStream.close();

		// Now parse the file as binary data
		fileInputStream = new FileInputStream(address);
		DataInputStream dis = new DataInputStream(fileInputStream);

		// look for 4 lines (i.e.: the header) and discard them
		int numnewlines = 3;
		while (numnewlines > 0) {
			char c;
			do {
				c = (char) (dis.readUnsignedByte());
			} while (c != '\n');
			numnewlines--;
		}

		// read the image data
		double[][] data2D = new double[picHeight][picWidth];
		for (int row = 0; row < picHeight; row++) {
			for (int col = 0; col < picWidth; col++) {
				data2D[row][col] = dis.readUnsignedByte();
			}
		}

		return new Matrix(data2D);
	}

	// Convert Matrix to PGM with numbers of row and column
}
