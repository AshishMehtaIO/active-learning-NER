# active-learning-NER

Baseline

Steps to run the code:

	1. Add the path to the dataset in cell 2, lines 1, 2, 3
	
	2. Run published_baseline.ipynb cell by cell
	
	3. An output file 'results_of_active_learning.txt' will get created once the notebook has finished running
	
	4. Pass this file as input to score.py to generate the Area under the Learning Curve metric 
	

Evaluation metric = Area under the Learning Curve

Area under the learning curve for random sampling = 0.4741 (Baseline from milestone 2)

Area under the learning curve for active learning = 0.4889

The final graph containing the learning curves of both random sampling and active learning method is 'published_baseline_plot.png'
