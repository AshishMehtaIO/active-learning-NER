Baseline - We developed NER models which is trained on datasets containing 2, 4, 8, 16... sentences and determined the f1 score for each of the models on the test dataset. We plotted the learning curve which will serve as the baseline for this project. The smaller training datasets were created by randomly sampling from the training dataset. The objective of the project is to use active learning to beat the f1 scores which were obtained using the randomly sampled datasets.

The simple-baseline.py has two outputs. 
	1. plot of the learning curve
	2. text file with two columns, training dataset size and f1 scores.

The plot is named 'simple-baseline-plot.png'. The text file, 'simple-baseline-output.txt' is passed to the evaluation script. 

The evaluation metric is area under the learning curve (ALC). The ALC for the models trained on the randomly sampled datasets is 0.498.