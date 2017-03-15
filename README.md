## Introduction [![Build Status](https://travis-ci.org/wihoho/FaceRecognition.svg?branch=master)](https://travis-ci.org/wihoho/FaceRecognition)

In this project, PCA, LDA and LPP are successfully implemented in Java for face recognition. After the system is trained by the training data, the feature space “eigenfaces” through PCA, the feature space “fisherfaces” through LDA and the feature space “laplacianfaces” through LPP are found using respective methods. Later in this report, W is used to represent the obtained feature space. Once W is obtained, training faces are projected to subspace defined by W to construct FaceDB. When an unknown face is needed to recognize, this test face is firstly projected onto subspace W. Afterward, the program finds the K nearest neighbors of the projected data in FaceDB. Finally, the class label is assigned to the test face according to the majority vote among the neighbors. This classification algorithm is known as K-nearest neighbor. 

The below figure shows respective feature space:
![](https://lh5.googleusercontent.com/-KtrqHFBv7l8/UV1tYE4zvtI/AAAAAAAAA24/Bf8x6b3UER8/s730/Eigenfaces.jpg)

## Design
Because of the limitation of Markdown, I provide [the pdf document](https://dl.dropboxusercontent.com/u/37572555/Github/Face%20Recognition/FaceRecognition.pdf) for your reference.

Presentation: [https://www.dropbox.com/s/bawrbgx78kin9xf/Face%20Recognition%20Demo.pdf](https://dl.dropboxusercontent.com/u/37572555/Github/Face%20Recognition/Face%20Recognition%20Demo.pdf)


## Usage
As many people asked me about this project, I decided to revamp this project into a maven project and release maven dependency to make this project easier to be used by
others. In order to use this library, this first step is to add the below dependency.

    <dependency>
      <groupId>com.github.wihoho</groupId>
      <artifactId>face-recognition</artifactId>
      <version>1.0</version>
    </dependency>

After that, you may refer to <code>com.github.wihoho.TrainerTest</code> as below on the usage of the API.

    // Build a trainer
    Trainer trainer = Trainer.builder()
            .metric(new CosineDissimilarity())
            .featureType(FeatureType.PCA)
            .numberOfComponents(3)
            .k(1)
            .build();

    ...

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

## Contact
I am open to collaboration in any forms. Kindly contact me with below email.
* wihoho@gmail.com

## Acknowledgement
[1] Delac, K., Grgic, M., & Grgic, S. (2005). Independent comparative study of PCA, ICA, and LDA on the FERET data set. International Journal of Imaging Systems and Technology, 15(5), 252-260.  
[2] Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.  
[3] Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). Eigenfaces vs. fisherfaces: Recognition using class specific linear projection. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 19(7), 711-720.  
[4]  He, X., Yan, S., Hu, Y., Niyogi, P., & Zhang, H. J. (2005). Face recognition using laplacianfaces. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 27(3), 328-340.  
[5] bytefish, awesome project, [https://github.com/bytefish/facerec.git](https://github.com/bytefish/facerec.git)  
[6] ORL Database of Faces, [http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)


## License
MIT License

Copyright © 2016 wihoho <wihoho@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.