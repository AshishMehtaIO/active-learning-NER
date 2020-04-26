The final metric that we report is the **Area under the Learning Curce (ALC)** metric and a **normalized global score**. For active learning methods, we plot the F-score of the respective active learning strategy as a function of the number of labels required [1]. For the implemented baseline of random strategy, the curve looks like follows. 

 	![Baseline FScore plot](baseline_plot.png)

The area under this learning curve also called as the ALC, is used as a global metric to evalute the final performance. 

A normalized Global Score is defined to rank various active learning strategies.  We consider two baseline learning curves:
The ideal learning curve, obtained when perfect predictions are made. It goes up vertically then follows ALC=1 horizontally. It has the maximum area "Amax". The "lazy" learning curve, obtained by making random predictions. It follows a straight horizontal line. We call its area "Arand". We can then define a normalized global score as follows: 

*globalscore= (ALC−Arand)/(Amax−Arand)*

The higher the ALC the better the active learning strategy. We hope to produce active learning strategies which have a higher ALC compared to this random baseline. The random baseline is expected to have a gloabal score of 0.0. All active learing strategies should produce better resuslts and have a positive global score if they work better than just the random baseline.

The AUC metric and the global socre were first introduced in the 2010 Active Learning Challenge [2, 3]. It has been used in various Active Learning research including Active Learning on NER [4].

---
To run the evaluation, first run the baseline model using:
`python simple_baseline.py`
This will train the model and geenrate the plot of F-score versus number of samples using the baseline strategy. It creates a file called `input_for_evaluation_script.txt` that contains the data for the scoring script. The scoring script can then be used to generate the score for the baseline using:
`python score.py input_for_evaluation_script.txt`
The .txt file can potentially be replaced with any other txt file to evaluate other methods.

The output geenrated by the scoring file is as follows:
ALC for baseline:  0.4987978168047594
Global score for baseline:  0.0

[1] Arora, Shilpa, and Sachin Agarwal. "Active learning for natural language processing." Language Technologies Institute School of Computer Science Carnegie Mellon University (2007).
[2] http://www.causality.inf.ethz.ch/activelearning.php?page=evaluation#cont
[3] Guyon, Isabelle, et al. "Results of the active learning challenge." Active Learning and Experimental Design workshop In conjunction with AISTATS 2010. 2011.
[4] Chen, Yukun, et al. "A study of active learning methods for named entity recognition in clinical text." Journal of biomedical informatics 58 (2015): 11-18.
