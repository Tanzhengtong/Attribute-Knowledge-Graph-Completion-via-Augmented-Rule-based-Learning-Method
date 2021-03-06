# Attribute Knowledge Graph Completion via Augmented Rule-based Learning Method
The augmented rule-based learning system is built on [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/) and is also called ***Augmented AnyBURL***. The complete source code of the Augmented Rule-based Learning System is stored in [Google Drive](https://drive.google.com/drive/folders/1Dypoy9xSNmGs7LjEm0sT1JPZ_9KGXTZo?usp=sharing). To run ***Augmented AnyBURL***, there are five steps to go.
### Step 1: Locate the working directory
Before you start to run the Augmented AnyBURL, it is important for you to go to the working directory first since all the modules are here. The working directory is ***'/AnyBURL/de/unima/ki/anyburl'***. There are some explanations of the files in the working directory:\
\
***rules***: This folder contains the rules learned from AnyBURL and the augmented rules from Augmented module. In additionally, this folder also store the previous results on different data sets. If you want to test the step 4 and step5 directly, you can copy the **rule files** (e.g. 'alpha-10', 'alpha-100', etc) and ***augmented rule file*** (i.e., 'augmented_rule') to the folder ***rules***.\
***data 3***: This folfer contains the data setes used in Augmented AnyBURL.\
***predictions***: This folder contains the results of link prediciton tasks from Step 4.\
***Previous Version of Augmentation Module***: This folder contains the previous versions of augmented modules.\
***Augmentation_module_V8.py*** : The latest version of Augmentation module.\
***AnyBURL.jar***: AnyBURL is packaged as jar file and requires no external resources. More information about [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/).\
***config-learn.properties***: The configuration file for rule learning task (Step 2).\
***config-apply.properties***: The configuration file for link prediction tasks (Step 4).\
***config-eval.properties***: The configuration file for Link prediction evaluation (Step 5).
## Step 2: Learning
### Input:
The data set in folder ***data 3*** (e.g. 'family', 'FB15-237', etc)
### Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.LearnReinforced config-learn.properties
### Comment:
Open the ***config-learn.properties*** file and modify the line that points to the training file to select the dataset you want to apply to AnyBURL. There are already some data sets to select where the ***FB15K*** and ***FB24K*** are the Attribute Knowledge Graphs. This step will create the output folder rule and store the CP rules in the files.
### Sample output:
350	2	0.005714285714285714	brother(212,Y) <= niece(A,Y)
Where the first number is the number of triplet sets that satisfy the rule body pattern, the second number is the number of triplet sets that satisfy both the rule body pattern and rule head pattern, the third number is the standard confidence, and the last column refers to the rule. 
### Output
The rule files (e.g. 'alpha-10', 'alpha-100', etc) in folder ***rule***
## Step 3: Rule augmentation
### Input:
The data set in folder ***data 3*** (e.g. 'family', 'FB15-237', etc) and the rule files (e.g. 'alpha-10', 'alpha-100', etc) in folder ***rule***
### Command: 
Python Augmentation_module_V8.py Family 0 0.8 0.05 4 50 1
### Comment: 
To run the augmentation module, you may be required to install some Python packages first, In addition, you need to specify some hyper-parameters in the command: the first parameter is the dataset you choose (e.g., Family), the second parameter the ***MinSC***, and the third one is the ***MaxSC***. The fourth one is the threshold of head coverage ***MinHC***, the fifth one is the support threshold ***MinSupport***, the sixth parameter is the runtime of AnyBURL, and the last one is the boolean value that indicates if the KG is an attribute knowledge graph.
### Comment: 
This module will enhance each rule from rule sets and create the output file augmented_rule and store new facts with corresponding original rule and new SC.
### Sample output:
108	21	0.1944	uncle(X,Y) <= uncle(X,A), brother(A,B), brother(Y,B)	710	uncle	698	0.9538\
Where the first four columns show the information about the original rule, the fifth to seven columns indicate the new fact inferred from the attribute rule, and the last column is the new SC of the attribute rule.
### Output:
The augmented rule file ***augmented_rule*** in folder ***rule***
## Step 4: Predicting
### Input:
The data set in folder ***data 3*** (e.g. 'family', 'FB15-237', etc), the rule files (e.g. 'alpha-10', 'alpha-100', etc) in folder ***rule***, and the augmented rule file ***augmented_rule***(optional) in folder ***rule***
### Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply config-apply.properties
### Comment:
Open the ***config-apply.properties*** file and change the hyper-parameters if you need. Note that you must specify the rules you have previously learned from step1, and you need to specify the file storing augmented rules you have obtained from step 3 with the line ***PATH_ARGMENTED_RULES=rules/augmented_rule***.
### Sample output:
16 niece 227
Heads: 16	0.3125	15	0.2429	598	0.2374???\
Tails: 41	0.328125	10	0.3125	227	0.3125	604	0.2129???\
Where the first row indicates the exact link prediction task, the following section started from Head is the candidates of head entity prediction results where 16 is the entity id and 0.3125 is the highest ***SC*** of the rule that suggests entity 16. The last section started from Tail is the candidates of tail entity prediction results where 41 is the entity id, and 0.328125 is the highest ***SC*** of the rule that infers entity 41.
### Output:
The prediction result files (e.g. 'alpha-10', 'alpha-100', etc) in folder ***prediction***
## Step 5: Link prediction evaluation
### Input:
The data set in folder ***data 3*** (e.g. 'family', 'FB15-237', etc) and the prediction result files (e.g. 'alpha-10', 'alpha-100', etc) in folder ***prediction***
### Command:
java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Eval config-eval.properties  
### Comment:
To evaluate the link prediction results, use this command after modifying ***config-eval.properties*** (if necessary). The evaluation results are printed to the standard output.
### Sample output:
10   0.3196   0.8106   0.4796\
50   0.3262   0.8115   0.4833\
100   0.3266   0.8118   0.4835\
1000   0.3254   0.8117   0.4826\
Where the first column refers to the time used for learning, the second column shows the ***hits@1*** score, and the third column the ***hits@10 score***. The last column is the ***MRR*** (approximated, based on the top-k only).
### Output:
The link prediction evaluation results
