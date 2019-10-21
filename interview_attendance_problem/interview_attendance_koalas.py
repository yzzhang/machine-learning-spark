import numpy as np
import pandas as pd
import databricks.koalas as ks
import matplotlib.pyplot as plt
import matplotlib as mpl
from   datetime import datetime
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def transformColumn(column_values, func, func_type):
    '''
    This function is to transform a given column (column_values) of a Koalas DataFrame or Series.
    This function is needed because the current Koalas requires that the applied function has 
    an explictly specified return type. Because of this, we cannot use lambda function directly 
    since lambda function does not have an explicit return type.
    '''
    def transform_column(column_element) -> func_type:
        return func(column_element)
    
    cvalues = column_values
        
    cvalues = cvalues.apply(transform_column)
            
    return cvalues

class OneHotEncodeData(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        This class is to one-hot encode the categorical features.
        '''
        self.one_hot_feature_names = ['Client name', 
                        'Industry', 
                        'Location', 
                        'Position to be closed', 
                        'Nature of Skillset',
                        'Interview Type', 
                        #'Name(Cand ID)', 
                        'Gender', 
                        'Candidate Current Location',
                        'Candidate Job Location', 
                        'Interview Venue', 
                        'Candidate Native location',
                        'Have you obtained the necessary permission to start at the required time',
                        'Hope there will be no unscheduled meetings',
                        'Can I Call you three hours before the interview and follow up on your attendance for the interview',
                        'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
                        'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
                        'Are you clear with the venue details and the landmark.',
                        'Has the call letter been shared', 
                        'Marital Status']
        self.label_encoders   = None
        self.one_hot_encoders = None
        
    def fit(self, X, y=None):       
        return self
    
    def transform(self, X, y=None):  
        X1 = X.copy()
        X1 = ks.get_dummies(X1)
        return X1
    
class FeaturesUppercase(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, drop_feature_names):
        '''
        This class is to change feature values to uppercase.
        '''
        self.feature_names      = feature_names
        self.drop_feature_names = drop_feature_names
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        '''
        This method is to change feature values to uppercase.
        '''
        
        func = lambda x: x.strip().upper()
        
        #def transform_column(column_element) -> str:
        #    return func(column_element)
        
        X_uppercase = X.copy()
        
        for fname in self.feature_names:
            values = X_uppercase[fname]
            values = values.fillna('NaN')
            # values = values.apply(transform_column)
            values = transformColumn(values, func, str)
            X_uppercase[fname] = values
        
        # drop less important features
        X_uppercase = X_uppercase.drop(self.drop_feature_names, axis=1)
            
        return X_uppercase   
    
class ParseInterviewDate(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        This class is to splits the date of interview into day (2 digits), month (2 digits), year (4 digits).
        '''     
    def __parseDate(self, string, delimit):
        try:
            if ('&' in string):
                subs = tuple(string.split('&'))
                string = subs[0]
        except:
            print ('TypeError: {}'.format(string))
            return None
        
        string = string.strip()
        
        try:
            d = datetime.strptime(string, '%d{0}%m{0}%Y'.format(delimit))
        except:
            try:
                d = datetime.strptime(string, '%d{0}%m{0}%y'.format(delimit))
            except:
                try:
                     d = datetime.strptime(string, '%d{0}%b{0}%Y'.format(delimit))
                except:
                    try:
                         d = datetime.strptime(string, '%d{0}%b{0}%y'.format(delimit))
                    except:
                        try:
                            d = datetime.strptime(string, '%b{0}%d{0}%Y'.format(delimit))
                        except:
                            try:
                                d = datetime.strptime(string, '%b{0}%d{0}%y'.format(delimit))
                            except:
                                d = None
        return d
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        '''
        This method splits the date of interview into day (2 digits), month (2 digits), year (4 digits).
        '''
        
        def transform_date(ditem):
            if (isinstance(ditem, str) and len(ditem) > 0):
                if ('.' in ditem):
                    d = self.__parseDate(ditem, '.')
                elif ('/' in ditem):
                    d = self.__parseDate(ditem, '/')
                elif ('-' in ditem):
                    d = self.__parseDate(ditem, '-')
                elif (' ' in ditem):
                    d = self.__parseDate(ditem, ' ')
                else:
                    d = None
                    
                if (d is None):
                    return 0, 0, 0
                else:
                    return d.day, d.month, d.year
                
        def get_day(column_element) -> int:
            try:
                day, month, year = transform_date(column_element)
                return int(day)
            except:
                return 0
        
        def get_month(column_element) -> int:
            try:
                day, month, year = transform_date(column_element)
                return int(month)
            except:
                return 0
        
        def get_year(column_element) -> int:
            try:
                day, month, year = transform_date(column_element)
                return int(year)
            except:
                return 0
        
        
        X1 = X.copy()
        
        X1['Year'] = X1['Date of Interview']
        X1['Month'] = X1['Date of Interview']
        X1['Day'] = X1['Date of Interview']
        
        func_map = {'Year' : get_year, 'Month' : get_month, 'Day' : get_day}
        for cname in func_map:
            cvalue = X1[cname]
            cvalue = cvalue.apply(func_map[cname])
            X1[cname] = cvalue
         
        return X1 
    
class BucketSkillset(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        This class is to re-bucket the skill sets and candidates location features 
        to combine small catogaries into one catogary 'Others'.
        '''
        self.skillset = ['JAVA/J2EE/Struts/Hibernate', 'Fresher', 'Accounting Operations', 'CDD KYC', 'Routine', 'Oracle', 
          'JAVA/SPRING/HIBERNATE/JSF', 'Java J2EE', 'SAS', 'Oracle Plsql', 'Java Developer', 
          'Lending and Liabilities', 'Banking Operations', 'Java', 'Core Java', 'Java J2ee', 'T-24 developer', 
          'Senior software engineer-Mednet', 'ALS Testing', 'SCCM', 'COTS Developer', 'Analytical R & D', 
          'Sr Automation Testing', 'Regulatory', 'Hadoop', 'testing', 'Java', 'ETL', 'Publishing']       
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        '''
        This method is to re-bucket the skill sets features.
        '''
        func = lambda x: x if x in self.skillset else 'Others'
               
        X1 = X.copy()
        
        cname = 'Nature of Skillset'
        cvalue = X1[cname]
        cvalue = transformColumn(cvalue, func, str)
        X1[cname] = cvalue
            
        return X1  
    
class BucketLocation(BaseEstimator, TransformerMixin):
    def __init__(self):
        '''
        This class is to re-bucket the candidates location features 
        to combine small catogaries into one catogary 'Others'.
        '''
        
        self.candidate_locations = ['Chennai', 'Hyderabad', 'Bangalore', 'Gurgaon', 'Cuttack', 'Cochin', 
                          'Pune', 'Coimbatore', 'Allahabad', 'Noida', 'Visakapatinam', 'Nagercoil',
                          'Trivandrum', 'Kolkata', 'Trichy', 'Vellore']
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        '''
        This method is to re-bucket the candidates native locations features.
        '''
            
        X1 = X.copy()
        
        func = lambda x: x if x in self.candidate_locations else 'Others'
        
        cname = 'Candidate Native location'
        cvalue = X1[cname]
        cvalue = transformColumn(cvalue, func, str)
        X1[cname] = cvalue
            
        return X1  
    
class GridSearch(object):
    def __init__(self, cv=10):
        '''
        This class finds the best model via Grid Search.
        '''
        self.grid_param = [
            {'n_estimators': range(68,69), # range(60, 70) # best 68
             'max_depth'   : range(8,9)}   # range(5, 10)}  # best 8
        ]
        self.cv = cv
        self.scoring_function = make_scorer(f1_score, greater_is_better=True) 
        self.gridSearch = None
        
    def fit(self, X, y):
        rfc = RandomForestClassifier()
        self.gridSearch = GridSearchCV(rfc, self.grid_param, cv=self.cv, scoring=self.scoring_function)
        self.gridSearch.fit(X, y)
        return self.gridSearch.best_estimator_
    
class PredictInterview(object):
    def __init__(self):
        '''
        This class is to predict the probability of a candidate attending scheduled interviews.
        '''
        self.dataset_file_name = 'Interview_Attendance_Data.csv'
        self.feature_names = ['Date of Interview', 
                       'Client name', 
                       'Industry', 
                       'Location', 
                       'Position to be closed', 
                       'Nature of Skillset',
                       'Interview Type', 
                       #'Name(Cand ID)',
                       'Gender', 
                       'Candidate Current Location',
                       'Candidate Job Location', 
                       'Interview Venue', 
                       'Candidate Native location',
                       'Have you obtained the necessary permission to start at the required time',
                       'Hope there will be no unscheduled meetings',
                       'Can I Call you three hours before the interview and follow up on your attendance for the interview',
                       'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
                       'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
                       'Are you clear with the venue details and the landmark.',
                       'Has the call letter been shared', 'Marital Status']
        
        self.drop_feature_names = [
                        'Name(Cand ID)',
                        'Date of Interview', 
                        '_c22',
                        '_c23',
                        '_c24',
                        '_c25',
                        '_c26']
        
        self.dataset = None
        self.rfc     = None
        self.gridSearch = None
        self.X_train = None
        self.y_train = None
        self.X_test  = None
        self.y_test  = None
        self.y_pred  = None
        self.X_clean = None
        self.y_clean = None
        self.X_train_encoded = None
        self.X_test_encoded  = None
        self.y_train_encoded = None
        self.accuracy_score  = None 
        self.f1_score        = None
        self.oneHotEncoder   = None
        self.X_test_name_ids = None
        self.pipeline = None
        
        
    def loadData(self, path=None):
        '''
        This method loads a dataset file as a Pandas DataFrame, assuming that the dataset file is in csv format.
        It also shuffles the loaded dataset as part of data preprocessing.
        '''
        if (path != None):
            path = os.path.join(path, self.dataset_file_name)
        else:
            path = self.dataset_file_name
            
        dataset = ks.read_csv(path)  
        
        # shuffle data 
        self.dataset = dataset.sample(frac=1.0) 
        
        return self.dataset     
    
    def PreprocessData(self):
        '''
        This method preprocesses the loaded dataset before applying one-hot encoding.
        '''
            
        y = self.dataset['Observed Attendance']      # extract labels y
        X = self.dataset.drop('Observed Attendance') # extract features X
        
        self.oneHotEncoder = OneHotEncodeData()
        
        self.pipeline = Pipeline([
            ('bucket_skillset', BucketSkillset()),
            ('bucket_location', BucketLocation()),
            ('parse_interview_date', ParseInterviewDate()),
            ('features_to_uppercase', FeaturesUppercase(self.feature_names, self.drop_feature_names)),
            ('one_hot_encoder', self.oneHotEncoder)
        ])
        
        X_1hot = self.pipeline.fit_transform(X)
        
        # fill up missing labels and then change labels to uppercase
        y = y.fillna('NaN')
        
        func = lambda x: x.strip().upper()
        
        y_uppercase = transformColumn(y, func, str) 
        
        # separate labeled records from unlabeled records
        self.X_train_encoded = X_1hot[y_uppercase != 'NAN']
        self.X_test_encoded  = X_1hot[y_uppercase == 'NAN']
        
        # save Names/ID for reporting later one
        self.X_test_name_ids = self.dataset['Name(Cand ID)'].loc[y_uppercase == 'NAN']
        
        y_train = y_uppercase.loc[y_uppercase != 'NAN']
        
        # encode labels as follows: 0 - NO, 1 - YES, NAN - NAN
        func = lambda x: 1 if x == 'YES' else 0
        
        y = transformColumn(y_train, func, int)
        
        self.y_train_encoded = y
        
        self.X_clean = X_1hot
        self.y_clean = y_uppercase
        
        return None
    
    def __splitData(self):
        '''
        This method triggers data preprocsssing and split dataset into training and testing datasets.
        '''
        X_train_encoded = self.X_train_encoded.to_numpy()
        y_train_encoded = self.y_train_encoded.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_encoded, 
                                                                                y_train_encoded, 
                                                                                test_size = 0.25, random_state = 0)
        return (self.X_train, self.X_test, self.y_train, self.y_test)
    
    def trainModel(self):
        '''
        This method triggers splitting dataset and then find a best RandomForest model via grid search 
        using the training features and labels.
        '''
        X_train, X_test, y_train, y_test = self.__splitData()
        self.gridSearch = GridSearch()
        self.rfc = self.gridSearch.fit(X_train, y_train)
        return self.rfc
    
    def predictClasses(self):
        '''
        This method predicts classes (YES or NO) using a trained model.
        '''
        if (self.rfc is None):
            print("No trained model available, please train a model first!")
            return None
        
        self.y_pred = self.rfc.predict(self.X_test)
        return self.y_pred
    
    def getModelMetrics(self):
        '''
        This method obtains the class prediction scores: (Accuracy Score, R2, F1).
        '''
        if (self.y_test is None or self.y_pred is None):
            print('Failed to get model performance metrics because y_test is null or y_pred is null!')
            return None
        
        self.accuracy_score = accuracy_score(self.y_test, self.y_pred)
        self.f1_score = f1_score(self.y_test, self.y_pred)
        
        pred = self.predictAttendanceProbability(self.X_test)[:, 1]
        actual = self.y_test.astype(float)
        
        self.rmse_score = np.sqrt(mean_squared_error(actual, pred))
        self.mae_score = mean_absolute_error(actual, pred)
        self.r2_score = r2_score(actual, pred)
        
        return (self.accuracy_score, self.f1_score, self.rmse_score, self.mae_score, self.r2_score)
    
    def predictNullAttendanceProbability(self):
        '''
        This method uses a trained model to predict the attendance probability for 
        the candidates where the "Observed Attendance" column is null.
        '''
        y_pred = self.rfc.predict_proba(self.X_test_encoded.to_numpy())
        return y_pred
    
    def predictNullAttendanceClasses(self):
        '''
        This method predicts classes (YES or NO) using a trained model for unlabeled data records.
        '''
        y_pred = self.rfc.predict(self.X_test_encoded.to_numpy())
        return y_pred
    
    def predictAttendanceProbability(self, X):
        '''
        Given one preprocessed (including one-hot encoding) data smaple X,
        this method returns the probability of attendance probability.
        '''
        y_pred = self.rfc.predict_proba(X)
        return y_pred
    
    def predictAttendanceClass(self, X):
        '''
        Given one preprocessed (including one-hot encoding) data smaple X,
        this method returns the attendance Yes/No.
        '''
        y_pred = self.rfc.predict(X)
        return y_pred
    
    def mlFlow(self):
        '''
        Training model in mlflow
        * https://www.mlflow.org/docs/latest/tutorial.html
        '''
        np.random.seed(40)
        with mlflow.start_run():
            self.loadData()
            self.PreprocessData()
            self.trainModel()
            self.predictClasses()
            accuracy_score, f1_score, rmse_score, mae_score, r2_score = self.getModelMetrics()

            print("Random Forest model:")
            print("  RMSE: {}".format(rmse_score))
            print("  MAE: {}".format(mae_score))
            print("  R2: {}".format(r2_score))
            print("Accuracy Score: {}".format(accuracy_score))
            print("  f1: {}".format(f1_score))

            mlflow.log_metric("rmse", rmse_score)
            mlflow.log_metric("r2", r2_score)
            mlflow.log_metric("mae", mae_score)
            mlflow.log_metric("accuracy", accuracy_score)
            mlflow.log_metric("f1", f1_score)

            mlflow.sklearn.log_model(self.rfc, "random_forest_model")
            
if __name__ == "__main__":
    predictInterview = PredictInterview()
    
    print('start mlflow ...')
    predictInterview.mlFlow()
    
    print('process null attendance ...')
    pred_probs   = predictInterview.predictNullAttendanceProbability()
    pred_classes = predictInterview.predictNullAttendanceClasses()

    x = predictInterview.X_test_name_ids.to_numpy() 
    z = zip(x, pred_probs, pred_classes)
    answers = ('no', 'yes')

    result = [[x1, p1[1], answers[c]] for x1, p1, c in z]
    result_df = pd.DataFrame(np.array(result), columns=['Names/ID', 'Probability', 'Yes/No'])
    
    print('output interview_prediction.csv ...')
    result_df.to_csv('interview_prediction.csv')
    
    print('all done')