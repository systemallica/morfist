# morfist: mixed-output-rf
Multi-target Random Forest implementation that can mix both classification and regression tasks.

Morfist implements the Random Forest algorithm (Breiman, 2001)
with support for mixed-task multi-task learning, i.e., it is possible to train the model on any number
of classification tasks and regression tasks, simultaneously. Morfist's mixed multi-task learning implementation follows that proposed by Linusson (2013). 

* [Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32](https://link.springer.com/article/10.1023%2FA%3A1010933404324).
* [Linusson, H. (2013). Multi-output random forests](https://pdfs.semanticscholar.org/4219/f87ed41c558d43cf78f63976cf87bcd7ebb0.pdf).

## Installation

With pip:
```
pip install decision-tree-morfist
```
With conda:
```
conda install -c systemallica decision-tree-morfist
```
## Usage

### Initialising the model

- Similarly to a scikit-learn [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), a MixedRandomForest can be initialised in this way:
```
from morfist import MixedRandomForest

mrf = MixedRandomForest(
    n_estimators=n_trees,
    min_samples_leaf=1,
    classification_targets=[0]
)
```
- The available parameters are:
    - **n_estimators(int)**: the number of trees in the forest. Optional. Default value: 10.
    
    - **max_features(int | float | str)**: the number of features to consider when looking for the best split. Optional. Default value: 'sqrt'.
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
        - If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
        - If “log2”, then max_features=log2(n_features).
        - If None, then max_features=n_features.
    
        Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
    
    - **min_samples_leaf(int)**: the minimum number of samples required to be at a leaf node. Optional. Default value: 5.
    
        Note: A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
        
    - **choose_split(str)**: method used to find the best split. Optional. Default value: 'mean'.
    
        By default, the mean information gain will be used.
        
        - Possible values:
            - 'mean': the mean information gain is used.
            - 'max': the maximum information gain is used.
        
    - **classification_targets(int[])**: features that are part of the classification task. Optional. Default value: None.
    
        If no classification_targets are specified, the random forest will treat all variables as regression variables.

### Training the model

- Once the model is initialised, it can be fitted like this:
    ```
    mrf.fit(X, y)
    ```
    Where X are the training examples and Y are their respective labels(if they are categorical) or values(if they are numerical)

### Prediction

- The model can be now used to predict new instances.
    - Class/value:
    ```
    mrf.predict(x)
    ```
    - Probability:
    ```
    mrf.predict_proba(x)
    ```
  
## TODO:
* Speed up the learning algorithm implementation (morfist is currently **much** slower than the Random Forest implementation available in scikit-learn) 
