from random import sample
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PseudoLabeller(BaseEstimator, TransformerMixin):
    '''
    Implementation of PseudoLabelling, a semi-supervised technique. 
    '''
    def __init__(self, model, features: list, target: list, labelled_data: pd.DataFrame, 
    unlabelled_data: pd.DataFrame, sample_rate = 0.2, shuffle = False, 
    shuffle_seed = 42) -> None:
        self.model = model      #ML model
        self.features = features    #Columns of input features
        self.target = target        #Columns of target features
        self.labelled_data = labelled_data  #Labelled dataset
        self.unlabelled_data = unlabelled_data  #Unlabelled dataset
        self.shuffle = shuffle          # add shuffle option later
        self.shuffle_seed = shuffle_seed
        self.input_features = self.labelled_data[self.features]
        self.output_target = self.labelled_data[self.target]
        self.sample_rate = sample_rate
        self.num_samples = int(len(self.unlabelled_data) * self.sample_rate)

    def _create_pseudo_labels_for_unlabelled_data(self):
        self.model.fit(self.input_features, self.output_target)
        unlabelled_data_copy = self.unlabelled_data.copy()
        pseudo_labels = self.model.predict(unlabelled_data_copy[self.features])
        unlabelled_data_copy[self.target] = pseudo_labels
        sampled_unlabelled_pseudo_data = unlabelled_data_copy.sample(self.num_samples, 
        self.shuffle_seed)
        # append sample unlabeled with pseudo labels with original training set
        train_set = self.labelled_data.copy()
        pseudo_set = pd.concat(sampled_unlabelled_pseudo_data, train_set, axis = 1)
        return pseudo_set
    
    def fit(self):
        data_with_pseudo_label = self._create_pseudo_labels_for_unlabelled_data()
        self.model.fit(
            data_with_pseudo_label[self.features],
            data_with_pseudo_label[self.target]
        )

        return self
    
    def get_params(self, deep=True):
        return {
            "Model": self.model,
            "Features": self.features,
            "Target": self.target,
            "Sample rate": self.sample_rate,
            "Length of unlabelled Data": len(self.unlabelled_data),
            "Length of labelled Data": len(self.labelled_data),
            "Shuffle seed": self.shuffle_seed
        }
        

"""
TODO
1. Add probability threshold when adding data from the unlabelled set to
the new training set.
2. Add exception handling and improve code
"""