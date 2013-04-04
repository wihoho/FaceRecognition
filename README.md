##Introduction

In this project, PCA, LDA and LPP are successfully implemented in Java for face recognition. After the system is trained by the training data, the feature space “eigenfaces” through PCA, the feature space “fisherfaces” through LDA and the feature space “laplacianfaces” through LPP are found using respective methods. Later in this report, W is used to represent the obtained feature space. Once W is obtained, training faces are projected to subspace defined by W to construct FaceDB. When an unknown face is needed to recognize, this test face is firstly projected onto subspace W. Afterward, the program finds the K nearest neighbors of the projected data in FaceDB. Finally, the class label is assigned to the test face according to the majority vote among the neighbors. This classification algorithm is known as K-nearest neighbor. 

The below figure shows respective feature space:
![](https://lh5.googleusercontent.com/-KtrqHFBv7l8/UV1tYE4zvtI/AAAAAAAAA24/Bf8x6b3UER8/s730/Eigenfaces.jpg)

##Design
Because of the limitation of Markdown, I provide [the pdf document](https://www.dropbox.com/s/pvnd20j5xdo5wg6/FaceRecognition.pdf) for your reference.

##Acknowledgement
[1] Delac, K., Grgic, M., & Grgic, S. (2005). Independent comparative study of PCA, ICA, and LDA on the FERET data set. International Journal of Imaging Systems and Technology, 15(5), 252-260.  
[2] Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.  
[3] Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). Eigenfaces vs. fisherfaces: Recognition using class specific linear projection. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 19(7), 711-720.  
[4]  He, X., Yan, S., Hu, Y., Niyogi, P., & Zhang, H. J. (2005). Face recognition using laplacianfaces. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 27(3), 328-340.  
[5] bytefish, awesome project, [https://github.com/wihoho/facerec.git](https://github.com/wihoho/facerec.git)


