cd code/complex/
echo "Complex fb15k_237"
python3 inference.py fb15k_237
cd ../../

cd code/complex/
echo "Complex wordnet18rr"
python3 inference.py wordnet18rr
cd ../../

# cd code/causal_lm/
# echo "Causal LM Subject Relation ? fb15k237"
# python3 inference.py fb15k237 false
# cd ../../

# cd code/causal_lm/
# echo "Causal LM Object Relation ? fb15k237"
# python3 inference.py fb15k237 true
# cd ../../

# cd code/causal_lm/
# echo "Causal LM Subject Relation ? wordnet18rr"
# python3 inference.py wordnet18rr false
# cd ../../

# cd code/causal_lm/
# echo "Causal LM Object Relation ? wordnet18rr"
# python3 inference.py wordnet18rr true
# cd ../../

# cd code/masked_lm/
# echo "Masked LM Subject Relation ? fb15k237"
# python3 inference.py fb15k237 object
# cd ../../

# cd code/masked_lm/
# echo "Masked LM ? Relation Object fb15k237"
# python3 inference.py fb15k237 subject
# cd ../../

# cd code/masked_lm/
# echo "Masked LM ? Relation ? fb15k237"
# python3 inference.py fb15k237 random
# cd ../../

# cd code/masked_lm/
# echo "Masked LM Subject Relation ? wordnet18rr"
# python3 inference.py wordnet18rr object
# cd ../../

# cd code/masked_lm/
# echo "Masked LM ? Relation Object wordnet18rr"
# python3 inference.py wordnet18rr subject
# cd ../../

# cd code/masked_lm/
# echo "Masked LM ? Relation ? wordnet18rr"
# python3 inference.py wordnet18rr random
# cd ../../