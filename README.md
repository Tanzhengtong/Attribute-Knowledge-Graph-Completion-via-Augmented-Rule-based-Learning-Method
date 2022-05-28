# Numeric-attribute-prediction-using-rule-learning-method-Knowledge-Graph

The augmented rule-based learning system is built on AnyBURL and is also called Augmented AnyBURL. To run Augmented AnyBURL, there are four steps to go.
Step 1: Learning
Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.LearnReinforced config-learn.properties
Comment:
Open the config-learn.properties file and modify the line that points to the training file to select the dataset  that you want to apply AnyBURL. There are already some data sets for you to selects where the FB15K and FB24K is the Attribute Knowledge Graphs. This step will create the output folder rule and store the CP rules in the files.
Sample output:
350	2	0.005714285714285714	brother(212,Y) <= niece(A,Y)
Where the first number is the number of triplet sets that satisfy the rule body pattern, the second number is the number of triplet sets that satisfy both the rule body pattern and rule head pattern, the third number is the standard confidence, and the last column refers to the rule. 
Step 2: Rule augmentation
Command: 
Python Augmentation_module_V8.py Family 0 0.8 0.05 4 50 1
Comment: Python Augmentation_module_V8.py Family 0 0.8 0.05 4 50 0
To run augmentation module, you need to specify some Hyper-parameters. The first parameter is the dataset you choose (e.g., Family), the second  parameter the MinSC and the third one is the MaxSC. The fourth one is the threshold of head coverage MinHC, the fifth one is the support threshold MinSupport, the sixth parameter is the runtime of AnyBURL, and the last one is the boolean value that indicate if the KG is an attribute knowledge graph.
Comment: This module will enhance each rule from rule sets and create the output file augmented_rule and store new facts with corresponding original rule and new SC.

Step 3: Predicting
Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply config-apply.properties
Comment:
Open the config-apply.properties file and change the hyper-parameters if you need. Note that you must specify the rules you have previously learned from step1 and you must specify the file storing augmented rules you have obtained from step 2 with the line PATH_ARGMENTED_RULES=rules/augmented_rule
Sample output:
16 niece 227
Heads: 16	0.3125	15	0.24299065420560748	598	0.23743306260336558…
Tails: 41	0.328125	10	0.312521875	227	0.3125	604	0.21296296296296297…
Where the first row indicate the exact link prediction task. The following section started from Head is the candidates of head entity prediction results where 16 is the entity id and 0.3125 is the highest SC of the rules that suggest entity 16. The The last section started from Tail is the candidates of tail entity prediction results where 41 is the entity id and 0.328125 is the highest SC of the rules that inferred entity 41.

Step 4: Link prediction evaluation
Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply config-apply.properties
Comment:
To evaluate the link prediction results, use this command after modifying config-eval.properties (if necessary). The evaluation results are printed to the standard output.
Sample output:
10   0.3196   0.8106   0.4796
50   0.3262   0.8115   0.4833
100   0.3266   0.8118   0.4835
1000   0.3254   0.8117   0.4826
Where the first column refers to the time used for learning, the second column shows the hits@1 score, the third column the hits@10 score. The last column is the MRR (approximated, as its based on the top-k only).
