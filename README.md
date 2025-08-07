# DIC-C2DH-HeLa

U-Net cell segmentation on the DIC-C2DH-HeLa dataset<sup>[[1]](https://celltrackingchallenge.net/2d-datasets/)</sup> of the Cell Tracking Challenge (CTC)<sup>[[2]](https://celltrackingchallenge.net/)</sup>.

<p align="center">
Figure 1: U-Net predictions for segment 01 of the training dataset
<br>
<img src="figures/train01.gif" alt="train01" width="500"/>
<br><br>
<!-- Figure 2: U-Net predictions for segment 02 of the training dataset
<br>
<img src="figures/train02.gif" alt="train02" width="500"/>
</p> -->

# Setup

Run the Bash script "setup" located in the bin folder. Optional arguments are:

- **--train**: Path to the training dataset directory with CTC subdirectory structure. Default is "DIC-C2DH-HeLa-Train";
- **--test**: Path to the test dataset directory with CTC subdirectory structure. Default is "DIC-C2DH-HeLa";
- **--fetch**: Download the DIC-C2DH-HeLa dataset from the Cell Tracking Challenge site. Uses the local dataset specified by the **--train** and **--test** flags if this option is not specified;
- **--help**: Display script help.

# Architecture and training

The original U-Net network architecture<sup>[[3]](https://doi.org/10.1007/978-3-319-24574-4_28)</sup> with an added zero padding step after each convolutional layer was implemented with Tensorflow. The binary cross-entropy loss with the distance-based pixel weight maps described in the original paper has also been implemented. Different data augmentation methods were explored, mainly geometric variations such as rotations, elastic deformation<sup>[[4]](https://doi.org/10.1109/ICDAR.2003.1227801)</sup> and grid distortion<sup>[[5]](https://doi.org/10.3390/info11020125)</sup>. To correct gray value variations and to avoid an extra step of data augmentation, I added a Contrast Limited Adaptive Histogram Equalization (CLAHE)<sup>[[6]](https://doi.org/10.1109/VBC.1990.109340)</sup> pre-processing step using OpenCV. There are no post-processing steps that come after the neural network.

Both the training and test dataset are composed of two segments of a video recording. The two training segments were used as a training-validation pair in a round-robin fashion (basically GroupKFold cross-validation). The test dataset lacks ground truth masks and obviously can't be used for training in a supervised learning setting like the one used in this experiment.

## Results

The out-of-fold results with the training subset were fairly good given that I'm using a pretty standard approach with relatively old algorithms. The idea is to use something like the Noisy Student method (semi-supervised learning) + gray-scale variations (augmentation) in a future setting and see if it improves the results, I just need to find enough spare time for that.

- Average out-of-fold IoU: 0.866305
- Average out-of-fold dice loss: 0.117310

**I'm yet to send the model to the CTC organizers for evaluation on the test subset**, so I'm keeping the test predictions hidden for now (even though the algorithm is public). The two gifs in the header of this README are the predictions for both segments of the **training dataset**.

# References

[[1]](https://celltrackingchallenge.net/2d-datasets/) Data set provided by Dr. Gert van Cappellen Erasmus Medical Center. Rotterdam. The Netherlands.

[[2]](https://doi.org/10.1093/bioinformatics/btu080) Martin Maška, Vladimír Ulman, David Svoboda, Pavel Matula, Petr Matula, Cristina Ederra, Ainhoa Urbiola, Tomás España, Subramanian Venkatesan, Deepak M.W. Balak, Pavel Karas, Tereza Bolcková, Markéta Štreitová, Craig Carthel, Stefano Coraluppi, Nathalie Harder, Karl Rohr, Klas E. G. Magnusson, Joakim Jaldén, Helen M. Blau, Oleh Dzyubachyk, Pavel Křížek, Guy M. Hagen, David Pastor-Escuredo, Daniel Jimenez-Carretero, Maria J. Ledesma-Carbayo, Arrate Muñoz-Barrutia, Erik Meijering, Michal Kozubek, Carlos Ortiz-de-Solorzano, A benchmark for comparison of cell tracking algorithms, Bioinformatics, Volume 30, Issue 11, June 2014, Pages 1609–1617, doi: doi.org/10.1093/bioinformatics/btu080

[[3]](https://doi.org/10.1007/978-3-319-24574-4_28) Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351. Springer, Cham. doi.org/10.1007/978-3-319-24574-4_28

[[4]](https://doi.org/10.1109/ICDAR.2003.1227801) P. Y. Simard, D. Steinkraus and J. C. Platt, "Best practices for convolutional neural networks applied to visual document analysis," Seventh International Conference on Document Analysis and Recognition, 2003. Proceedings., Edinburgh, UK, 2003, pp. 958-963, doi: doi.org/10.1109/ICDAR.2003.1227801

[[5]](https://doi.org/10.3390/info11020125) Buslaev, A.; Iglovikov, V.I.; Khvedchenya, E.; Parinov, A.; Druzhinin, M.; Kalinin, A.A. Albumentations: Fast and Flexible Image Augmentations. Information 2020, 11, 125. doi.org/10.3390/info11020125

[[6]](https://doi.org/10.1109/VBC.1990.109340) S. M. Pizer, R. E. Johnston, J. P. Ericksen, B. C. Yankaskas and K. E. Muller, "Contrast-limited adaptive histogram equalization: speed and effectiveness," [1990] Proceedings of the First Conference on Visualization in Biomedical Computing, Atlanta, GA, USA, 1990, pp. 337-345, doi: doi.org/10.1109/VBC.1990.109340