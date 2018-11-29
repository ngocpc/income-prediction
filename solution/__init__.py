import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import category_encoders as ce

def get_pipeline():
    """
    This function should build an sklearn.pipeline.Pipeline object to train
    and evaluate a model on a pandas DataFrame. The pipeline should end with a
    custom Estimator that wraps a TensorFlow model. See the README for details.
    """
    
    extFeatures = FeatureExtracter()
    encodedFeatures = FeatueEncoder()
    
    # Grid search
    clf = GradientBoostingClassifier(random_state = 0)    
    grid_values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [2, 3, 4, 5]}
    grid_gb_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
    
    # Creat esitmators
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    
    # Use GB + Grid Search with features engineering
    estimators = [('encoder', ce.OneHotEncoder(cols=categorical_features)), ('clf', grid_gb_auc)]
    # estimators = [('encoder', ce.OneHotEncoder(cols=categorical_features)), ('clf', RandomForestClassifier(n_estimators=250, max_features=5))]
    
    # Creat a pipeline
    pipeline = Pipeline(estimators)
    
    return pipeline


class FeatureExtracter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):       
        return self
    
    def transform(self, X):  
        
        ts = TypeSelector(np.number)
        return ts.fit_transform(X)
    
class FeatueEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):        
        return self
    
    def transform(self, X):
            
        categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]        
        # Covert object features to categorical features
        for f in categorical_features:
            X[f] = X[f].astype("category")
          
        # Replace "?" to NaN         
        X.replace(" ?", np.nan, inplace = True)
        # print(X.head())    
                
        """
        # One-hot encoding with category_encoders
        ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
        X_impute_ohe = ohe.fit_transform(X_impute)
        X_impute_ohe.head()
        
        print("After\n")
        print("Dim: ", X_impute_ohe.shape)
        print(X_impute_ohe.head(10))
        
        # Replace the missing 'workclass' values by the most frequent 
        # X_copy["workclass"].replace(np.nan, "Private", inplace = True)
        """
        
        """
        # One-hot encoder with preprocessing
        enc = preprocessing.OneHotEncoder()                 
        lb_make = preprocessing.LabelEncoder()
        new_colnames = X_impute["workclass"].unique()
        #print(new_colnames)
        
        new_features = enc.fit_transform(X_impute["workclass"].values.reshape(-1,1)).toarray()
        dfOneHot = pd.DataFrame(new_features, columns = new_colnames)
        X_impute = pd.concat([X_impute, dfOneHot], axis=1)
        
        #X_impute = apply(X_impute, lb_make.fit_transform)
        #X_impute["workclass"] = enc.fit_transform(X_impute["workclass"])
        """
        """
        # Use one hot encoding technique to conver "education" to binary variables
        X_copy = pd.concat([X_copy, pd.get_dummies(X_copy['education'])], axis=1)
        X_copy.drop(['education'], axis = 1, inplace=True)
        
        # Use one hot encoding technique to conver "marital-status" to binary variables
        X_copy = pd.concat([X_copy, pd.get_dummies(X_copy['marital-status'])], axis=1)
        X_copy.drop(['marital-status'], axis = 1, inplace=True)
        
        # Use one hot encoding technique to conver "relationship" to binary variables
        X_copy = pd.concat([X_copy, pd.get_dummies(X_copy['relationship'])], axis=1)
        X_copy.drop(['relationship'], axis = 1, inplace=True)
        
        # Use one hot encoding technique to conver "race" to binary variables
        X_copy = pd.concat([X_copy, pd.get_dummies(X_copy['race'])], axis=1)
        X_copy.drop(['race'], axis = 1, inplace=True)
        
        # Use one hot encoding technique to conver "race" to binary variables
        X_copy = pd.concat([X_copy, pd.get_dummies(X_copy['sex'])], axis=1)
        X_copy.drop(['sex'], axis = 1, inplace=True)
        """
        #print("After\n")
        #print(X_impute.head(10))
        return X

class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])