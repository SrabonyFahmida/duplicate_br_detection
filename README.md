# duplicate_br_detection
1.data_split_80_20.py: This is the first step where we are splitting the dataset into train set (which is 80% data) and test set (which is 20% data). Data is splitted according to ground truth file.

2.GT_PlaceHolder.py: Combine and reformat data from two different sources; the bug reports and their associated duplicates (ground truth data). The output is structured in a way that each bug report key is linked to its duplicates

3.preprocess_data.py

4.Final_Ranking_updated.py : Evaluating Zero Shot Model on Test dataset.

5.fine_tune9.py : Fine Tuning the model on Train dataset.

6.fine_tune_final_ranking.py : Evaluating Fine Tuned model on Test dataset.


