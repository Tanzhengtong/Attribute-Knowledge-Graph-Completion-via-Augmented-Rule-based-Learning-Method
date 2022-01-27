
head_coverage=(0.01 0.05 0.1 0.2)
support=(2 5 10)
learning_time=(10 50 100)
for head_coverage in 0.01 0.05 0.1
do
    for support in 2 5 10
    do
        for learning_time in 10 50 100
        do  
            rm -f rules/augmented_rule-valid-test
            Python DT.py family 0.02 0.98 $head_coverage $support $learning_time
            java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Apply config-apply.properties
            java -Xmx3G -cp AnyBURL-JUNO.jar de.unima.ki.anyburl.Eval config-eval.properties
            echo Grid search current hyper-parameters $head_coverage $support $learning_time
            
        done
    done
done
