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

For more info on the possible parameters, visit the [documentation](https://systemallica.github.io/morfist/).

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
