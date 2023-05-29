import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

from colorama import Fore


class Regression:
    columns = None
    data = pd.DataFrame()  # the supplied dataframe
    data_cat = pd.DataFrame()  # the dataframe for categorical variables alone
    data_num = pd.DataFrame()  # the dataframe for numerical variables alone

    x = pd.DataFrame()
    y = pd.DataFrame()

    numerical_features = []  # names of numerical features
    categorical_features = []  # names of categorical features

    # the training and test sets
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    model_ = pd.DataFrame()    # the supplied model

    y_predicted = pd.DataFrame()  # the dataframe for the predicted y

    selected_features = []  # A list of the selected features
    reduced_selected_features = False  # whether the number of selected features was reduced

    cross_validation_scores = np.array([])  # gives the cross validation score for a model
    average_cv_score = None  # gives the average cross validation score for a model
    model_score = None  # gives the coefficient of determination score for a model

    def __init__(self, data__, columns=None):
        """
        initialises the regression object

        :param data__: The supplied dataset which is a dataframe.
        :param columns: The column names for the dataframe - Should be supplied if dataframe has no
                        column name or its column names are to be changed.

        """

        self.data = data__.copy(deep=True)
        columns = columns

        # if the column names are supplied, use it.
        if columns is not None:
            self.data.columns = columns

        self.selected_features = self.data.columns

    def initialDetails(self):
        """

        :return: returns a dictionary that contains the following:
                {   self.data.head(),
                    self.data.shape,
                    self.data.dtypes,
                    self.data.isna().any(),
                    self.data.columns,
                }
        """

        original_no_of_features = len(self.data.columns)

        for feature in self.data.columns:
            if self.data[feature].nunique() == 1:
                self.data = self.data.drop(labels=[feature], axis=1)

                # the original number of selected features has been reduced, i.e. features with constant values
                self.reduced_selected_features = True

        new_no_of_features = len(self.data.columns)

        if self.reduced_selected_features:
            print(Fore.BLUE + 'Note: ' + str(original_no_of_features - new_no_of_features) + ' of the supplied features'
                  + ' had a constant value and have been dropped. There are only ' + str(new_no_of_features) + ' of'
                  + ' the ' + str(original_no_of_features) + ' supplied features left',
                  Fore.RESET)

            # print(Fore.RESET)
        return {
            'head': self.data.head(),
            'data shape': self.data.shape,
            'data types': self.data.dtypes,
            'any missing data?': self.data.isna().any(),
            'columns': self.data.columns,
        }

    def specifyXY(self, x_list=None, y_list=None, x_indices=None, y_indices=None):
        """
        Specifies the index or indices for x and y in the supplied dataframe

        :param x_list: a list of the indices to be used with the iloc method in order to obtain self.x e.g  if
                        x_list = [1,2,3], then self.x = self.data.iloc[:, [1,2,3])
        :param y_list: a list of the indices to be used with the iloc method in order to obtain self.y as in x_list
                        above
        :param x_indices: a list with only two elements viz: the starting and the ending indices. e.g if
                            y_list = [6, 13], then self.x = self.data.iloc[6: 13]
        :param y_indices: a list with only two elements viz: the starting and the ending indices as in x_indices above.
        :return: returns the dimensions of x and y and the Variance Inflation Factors (VIF's) of the specified x and y.

        """

        try:
            self.x = (
                self.data.iloc[:, x_list] if x_list is not None  # using list
                else self.data.iloc[:, x_indices[0]:x_indices[-1]] if len(x_indices) > 1  # using indices
                else self.data.iloc[:, x_indices[0]] if len(x_indices) == 1  # using indices when there is 1 variable
                else None
            )

            self.y = (
                self.data.iloc[:, y_list] if y_list is not None
                else self.data.iloc[:, y_indices[0]:y_indices[-1]] if len(y_indices) > 1  # using indices
                else self.data.iloc[:, y_indices[0]] if len(y_indices) == 1  # using indices when there is 1 variable
                else None
            )

            print('x_shape', self.x.shape)
            print('y_shape', self.y.shape)

            vif_data = [vif(self.x.values, i) for i in range(len(self.x.columns))]

            return {
                'dimension of x =': self.x.shape,
                'dimension of y =': self.y.shape,
                'vif\'s': vif_data
             }

        except IndexError as error:
            return ('Error: ' + str(error).capitalize()
                    + '. Check the entered indices or lists of indices you have entered.')
        except TypeError as error:
            return 'Error: ' + str(error).capitalize() + '. The supplied list should only contain integers'

    def get_cat_dataframe(self):
        """

        :return: returns a dataframe containing only numerical features of the supplied or specified dataframe

        """
        return self.data_num

    def get_num_dataframe(self):
        """

        :return: returns a dataframe containing only categorical features of the supplied or specified dataframe

        """
        # returns a dataframe of numerical features
        return self.data_cat

    def featuresSplit(self):
        """
        splits the features (column) names to categorical and numerical columns

        :return: numerical feature names and  categorical features names
        """

        self.numerical_features = [
            feature for feature in self.data.columns
            if self.data[feature].dtype == np.int64 or self.data[feature].dtype == np.float64
        ]
        self.categorical_features = [feature for feature in self.data.columns if feature not in self.numerical_features]

        self.data_cat = self.data[self.categorical_features]
        self.data_num = self.data[self.numerical_features]

        return {'numerical_features': self.numerical_features, 'categorical_features': self.categorical_features}

    def get_data_corr(self):
        """

        :return: returns the correlations between the features of the supplied or specified dataframe

        """
        return self.data.corr() if not self.data.empty else None

    def get_train_corr(self):
        """

        :return: returns the correlations between the features of x_trains obtained from the specified dataframe

        """
        return self.x_train.corr() if not self.x_train.empty else None

    def plots(self):
        """
        Gives a plot of the target versus the features (numerical and categorical).
        :return: None
        """

        if self.y.empty or self.x.empty:
            print(Fore.LIGHTRED_EX, 'Note: You have to specify X and Y using specifyXY', Fore.RESET)

        else:
            for i in range(len(self.y.columns)):
                for feature in self.x.columns:
                    if feature in self.numerical_features:
                        plot = sns.scatterplot(x=feature, y=self.y.iloc[:, i], data=self.data_num)
                        plot.set(
                            title="Plot of " + feature.title() + ' against ' + self.y.iloc[:, i].name,
                            xlabel=plot.get_xlabel().title().replace('_', ' ').replace('.', ' '),
                            ylabel=plot.get_ylabel().title().replace('_', ' ').replace('.', ' ')
                        )
                        plt.savefig(
                            '../../cbm_images/' + feature + '_' + self.y.iloc[:, i].name.replace(' ', '_') + '.png'
                        )
                        plt.show()

                for feature in self.x.columns:
                    if feature in self.categorical_features:
                        plot = sns.stripplot(x=feature, y=self.y.iloc[:, i], data=self.data_cat)
                        plot.set(
                            title="Plot of " + feature.title() + ' against ' + self.y.iloc[:, i].name,
                            xlabel=plot.get_xlabel().title().replace('_', ' ').replace('.', ' '),
                            ylabel=plot.get_ylabel().title().replace('_', ' ').replace('.', ' ')
                        )
                        plt.savefig(
                            '../../cbm_images/' + feature + '_' + self.y.iloc[:, i].name.replace(' ', '_') + '.png'
                        )
                        plt.show()

    def trainTestSplit(self, test_size=None, train_size=None):
        """
        Splits the data into test and train sets.

        :param test_size: the size of the test dataset
        :param train_size: the size of the train dataset
        :return: the shapes of the x_train, x_test, y_train and y_test.
        """
        if test_size is None and train_size is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=123)
            print(Fore.RED, '\nWarning: Neither test_size nor train_size '
                            + 'was supplied and and so a test_size of 0.25 was used.', Fore.RESET)

        if test_size is not None and train_size is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=123)

        elif train_size is not None and test_size is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, train_size=train_size, random_state=123)

        else:
            # if both are supplied
            raise Exception('Both test_size and train_size were supplied. Supply only one')

        return {'x_train_shape': self.x_train.shape,
                'x_test_shape': self.x_test.shape,
                'y_train_shape': self.y_train.shape,
                'y_test_shape': self.y_test.shape}

    def grid_search_cv(self, estimator, parameters, metric_list=None, refit=None, no_of_features=None):
        """
        Performs a grid search (with cross validation scores) to find the best model across the supplied parameters


        :param estimator: the estimator to be used e.g Linear Regression
        :param parameters: A dictionary of the parameters to be used for the grid search
        :param metric_list: A list of metrics to be used for evaluating the models
        :param refit: the metric is to be used to select the best model
        :param no_of_features: the number of features to be used to build the models
        :return: returns GridSearchCV.best_estimator

        """
        # the grid_search_cv handles everything except that the parameters need to be set.
        if self.x_train.empty is None or self.x_train.empty:
            raise ValueError('You have not split the data. Use the trainTestSplit() method'
                             + ' and set the train_size or test_size. ')

        if refit is None and metric_list is not None:
            for metric in metric_list:
                refit = metric.__name__ if metric.__name__ == 'r2_score' else metric_list[0].__name__

        # uses a pipe that immediately fits the model without conducting feature selection (when no_of_features = 'all')
        pipe = Pipeline(steps=[
            ('model', estimator)
        ])

        # uses a pipe that does a model selection first and then feature selection
        if no_of_features is None or no_of_features != 'all':
            pipe = Pipeline(steps=[
                ('feature_selection',
                 RFE(
                     estimator=LinearRegression(),
                     n_features_to_select=(round(len(self.x_train.columns) / 2) if no_of_features is None
                                           else no_of_features),
                     step=1
                 )
                 ),
                ('model', estimator)
            ])

        scoring_dict = dict()
        if metric_list is not None:
            for metric in metric_list:
                scoring_dict[metric.__name__] = make_scorer(metric)

        models_to_search = GridSearchCV(
            pipe,
            param_grid=parameters,
            scoring=scoring_dict if scoring_dict is not None else None,
            refit=refit,
            cv=5
        )

        models_to_search.fit(self.x_train, self.y_train)

        # selecting the best model according to GridSearchCV
        best_estimator = None

        if no_of_features != 'all':
            best_estimator = models_to_search.best_estimator_[1]
        else:
            best_estimator = models_to_search.best_estimator_

        # obtaining the selected columns
        if no_of_features != 'all':
            truths_of_selection = pipe.named_steps['feature_selection'].get_support()  # each feature selected has
            # a 'True' in the truths_of_selection list

            self.selected_features = self.x_train.columns[truths_of_selection]

        else:
            self.selected_features = self.x_train.columns

        return_dict = {'best_estimator': best_estimator}
        return return_dict

    def model(self, estimator, metric_list=None, no_of_features=None, polynomial=False, poly_degree=None, save=False):
        """
            Builds a regression model using the supplied estimator

            :param estimator: the estimator to be used e.g Linear Regression
            :param polynomial: whether to used a polynomial regression of higher degree. Applicable if the estimator to be
                            used is Linear Regression.
            :param poly_degree: the degree of the polynomial regression eg. 2, 3, ...
            :param metric_list: A list of metrics to be used for evaluating the models
            :param no_of_features: the number of features to be used to build the models
            :param save: saves the model if save is True
            :return: returns the model fitted, and the scores obtained for the model using each supplied metric, one
                    for the cross validation data and one for the test data

         """
        # fits the required model
        if self.x_train.empty is None or self.x_train.empty:
            raise ValueError('You have not split the data. Use the trainTestSplit() method'
                             + ' and set the train_size or test_size. ')

        x_train_transformed = pd.DataFrame()
        x_test_transformed = pd.DataFrame()

        if poly_degree is None:
            poly_degree = 2

        if polynomial:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=True)
            x_train_transformed = poly.fit_transform(self.x_train)
            x_test_transformed = poly.fit_transform(self.x_test)

        # uses a pipe that immediately fits the model without conducting feature selection (when no_of_features = 'all')
        pipe = Pipeline(steps=[
                ('model', estimator)
            ])

        # uses a pipe that does a model selection first and then feature selection
        if no_of_features is None or no_of_features != 'all' and polynomial is False:
            pipe = Pipeline(steps=[
                ('feature_selection',
                 RFE(
                     estimator=LinearRegression(),
                     n_features_to_select=(round(len(self.x_train.columns) / 2)
                                           if no_of_features is None else no_of_features
                                           ),
                     step=1
                 )
                 )
                ,
                ('model', estimator)
            ])

        pipe.fit(self.x_train, self.y_train)

        if polynomial:
            pipe.fit(x_train_transformed, self.y_train)

        scoring_dict = dict()
        if metric_list is not None:
            for metric in metric_list:
                scoring_dict[metric.__name__] = make_scorer(metric)
                # print(scoring_dict)

        self.cross_validation_scores = cross_validate(
            pipe,
            X=self.x_train if polynomial is False else x_train_transformed,
            y=self.y_train,
            scoring=scoring_dict
        )

        # obtaining the selected columns
        if no_of_features != 'all':
            truths_of_selection = pipe.named_steps['feature_selection'].get_support()  # each feature selected has a 'True'
            # in the truths_of_selection list

            self.selected_features = self.x_train.columns[truths_of_selection]
            # print(selected_features)
        else:
            self.selected_features = self.x_train.columns

        # obtaining the accuracy score for the model
        self.model_score = (
            pipe.score(self.x_test, self.y_test) if polynomial is False
            else pipe.score(x_test_transformed, self.y_test)
        )

        # making prediction using the test data
        self.y_predicted = pipe.predict(self.x_test) if polynomial is False else pipe.predict(x_train_transformed)

        return_dict = {
            # if no_of_features == 'all, then there was no feature selection in pipe
            'model': pipe.steps[1][1] if no_of_features != 'all' else pipe.steps[0][1],
        }

        if metric_list is not None:
            for metric in metric_list:
                return_dict['cv_' + metric.__name__] = round(
                    self.cross_validation_scores['test_' + metric.__name__].mean(),
                    ndigits=4
                )

        if metric_list is not None:
            for metric in metric_list:
                return_dict['test_' + metric.__name__] = round(
                    metric(self.y_test, pipe.predict(self.x_test)),
                    ndigits=4
                ) if polynomial is False else round(
                    metric(self.y_test, pipe.predict(x_test_transformed)),
                    ndigits=4
                )

        # the model to be used for predicting
        self.model_ = pipe

        # saving the model
        if save:
            file_name = '../model/decision_tree.joblib'
            joblib.dump(self.model_, file_name)

        return return_dict

    def predict(self):
        """

        :return: a numpy array of the predicted result for the test data, obtained when a model is built.

        """
        return self.model_.predict(self.x_test)


def main():
    # Loading the data
    data = pd.read_csv('../data/data.txt', sep="   ", header=None, engine='python')
    print('Data Head')
    print(data.head())

    # The columns
    data.columns = [
        'Lever position (lp)',
        'Ship speed (v) [knots]',
        'Gas Turbine shaft torque (GTT) [kN m]',
        'Gas Turbine rate of revolutions (GTn) [rpm]',
        'Gas Generator rate of revolutions (GGn) [rpm]',
        'Starboard Propeller Torque (Ts) [kN]',
        'Port Propeller Torque (Tp) [kN]',
        'HP Turbine exit temperature (T48) [C]',
        'GT Compressor inlet air temperature (T1) [C]',
        'GT Compressor outlet air temperature (T2) [C]',
        'HP Turbine exit pressure (P48) [bar]',
        'GT Compressor inlet air pressure (P1) [bar]',
        'GT Compressor outlet air pressure (P2) [bar]',
        'Gas Turbine exhaust gas pressure (Pexh) [bar]',
        'Turbine Injection Control (TIC) [%]',
        'Fuel flow (mf) [kg/s]',
        'GT Compressor decay state coefficient',
        'GT Turbine decay state coefficient',
    ]

    # A Regression instance
    a = Regression(data)

    # Initial characteristics of the data
    initial_details = a.initialDetails()

    # the shape of the data
    print('data shape:')
    print(initial_details['data shape'])
    print()

    # the type of each column
    print('data types:')
    print(initial_details['data types'])
    print()

    # any missing data?
    print('any missing data:')
    print(initial_details['any missing data?'])
    print()

    # print(a.initialDetails()) # this can be used to get the original dictionary

    # Splitting the column name of features to categorical and numerical
    print(a.featuresSplit())

    # print(a.get_cat_dataframe())

    # Specifying which columns to use for the x and y [0] + list(range(2, 13))
    # The features Gas Turbine shaft torque (GTT) [kN m] and 'Gas Generator rate of revolutions (GGn) [rpm]' were used
    # because they had low VIF values, when put together

    print(a.specifyXY(x_list=[2, 4], y_list=[1, 14, 15],))

    # Plots of features versus targets
    a.plots()

    # train_test_split
    print('splitting to train and test datasets')
    print(a.trainTestSplit(train_size=0.23))

    # The Models
    # 1) Least Squares Regression Model
    print('\n')
    print('The Least Squares Regression')
    print(a.model(LinearRegression(), metric_list=[r2_score, mean_squared_error], no_of_features='all', polynomial=True))
    # print(a.model(Ridge(random_state=123), metric_list=[r2_score]))

    # 2) KNeighbors Regression
    # A grid search for the best K-Nearest Neighbour (KNN) Model
    print('\n')
    print('Searching for the Best KNeighbors Regression')
    parameters = {'model__n_neighbors': range(3, 30, 2)}
    print(a.grid_search_cv(estimator=KNeighborsRegressor(),
                           parameters=parameters,
                           metric_list=[r2_score, mean_squared_error],
                           no_of_features='all'))

    print('Using Best KNeighbors Regression Model')
    print(a.model(estimator=KNeighborsRegressor(n_neighbors=3),
                  metric_list=[r2_score, mean_squared_error],
                  no_of_features='all'))

    # 3) Support Vector Machines Regression (SVR)
    # A grid search for the best SVR
    print('\n')
    print('Searching for the Best Support Vector Regression')
    '''parameters = {'model__estimator__C': [10, 100, 1000, 10000, 100000, 1000000]}
    print(a.grid_search_cv(estimator=MultiOutputRegressor(estimator=SVR(max_iter=35000)),
                           parameters=parameters,
                           metric_list=[r2_score, mean_squared_error],
                           no_of_features='all'))'''

    print('Using the Best Support Vector Regression')
    print(a.model(estimator=MultiOutputRegressor(SVR(C=10)),
                  metric_list=[r2_score, mean_squared_error],
                  no_of_features='all'))

    # (4) Decision Tree (DT) Model
    # A grid search for the best Decision Tree Model'''
    print('\n')
    print('Searching for the Best Decision Tree Model')
    max_depth = range(100, 500, 100)
    max_features = [2]
    max_leaf_nodes = range(50, 500, 50)
    # decision_tree = [DecisionTreeRegressor(max_depth=md, max_features=mf, max_leaf_nodes=mln) for md, mf, mln in
    # dt_indices]

    parameters = {'model__max_depth': max_depth,
                  'model__max_features': max_features,
                  'model__max_leaf_nodes': max_leaf_nodes}

    print(a.grid_search_cv(estimator=DecisionTreeRegressor(),
                           parameters=parameters,
                           metric_list=[r2_score, mean_squared_error],
                           no_of_features='all'))

    print('Using the Best Decision Tree Model')
    print(a.model(estimator=DecisionTreeRegressor(max_depth=200,
                                                  max_features=2,
                                                  max_leaf_nodes=500,
                                                  random_state=123),
                  metric_list=[r2_score, mean_squared_error],
                  no_of_features='all', save=True))

    # printing the selected columns
    print()
    print("The selected features:")
    print(a.selected_features)

    # predicting performance decay using the decision tree model
    # print(a.predict())
    y_predicted = np.column_stack((a.predict(), a.y_test)).round(decimals=4)
    y_predicted = pd.DataFrame(y_predicted)

    # specifying the column names
    y_predicted.columns = [
        'Ship speed (v) [knots]_Predicted',
        'GT Compressor decay state coefficient_Predicted',
        'GT Turbine decay state coefficient_Predicted',
        'Ship speed (v) [knots]_Actual',
        'GT Compressor decay state coefficient_Actual',
        'GT Turbine decay state coefficient_Actual'
       ]
    print(y_predicted)

    # saving the prediction dataframe
    y_predicted.to_csv('../prediction/predicted_decay_of_performance.csv')


if __name__ == '__main__':
    main()
