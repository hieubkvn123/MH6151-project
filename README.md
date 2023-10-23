# MH6151-project
MH6151 Data Mining project

## 1. Set-up
- Install requirements with the following command:
```bash
pip install -r requirements.txt
```

## 2. Run the modelling files
- Run the python files with the format `modelling.py --model_name <model_name> --output_file <path>` and save output to the folder `./outputs`. For example, to run and save the output for random forest classifier, execute the following command:
```bash
python modelling.py --model_name random_forest --output_file outputs/random_forest.txt
```

- To add oversampling step to the training data, simply add the `--oversampling` option in the command.
```bash
python modelling.py --model_name random_forest --output_file outputs/random_forest.txt --oversampling
```

# References
- Random oversampling and undersampling for imbalanced classification : [Link](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/).
- AdaBoost Algorithm: Understand, Implement and Master AdaBoost : [Link](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/).
- AdaBoost clearly explained (Josh Starmer) : [Link (Youtube)](https://www.youtube.com/watch?v=LsK-xG1cLYA).
