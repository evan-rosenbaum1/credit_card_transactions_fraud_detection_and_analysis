import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelEvaluator:
    def __init__(self, model, param_grid, X, y, sampling_techniques, random_state=19):
        self.model = model
        self.param_grid = param_grid
        self.X = X
        self.y = y
        self.sampling_techniques = sampling_techniques
        self.random_state = random_state
        self.eval_results_df = None
        self.best_model = None
        self.best_model_params = None
        self.best_model_train_residuals = None
        self.best_model_test_residuals = None
        self.feature_importances_df = None
        self.boosting_results_df = None

    def get_preprocessor(self, X):
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop= 'first', sparse_output=False), categorical_features)
            ])

    def evaluate(self):
        # Create results list
        results = []
        
        # Dictionaries to store residuals
        self.best_model_train_residuals = {}
        self.best_model_test_residuals = {}

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=self.random_state)
        
        for name, sampler in self.sampling_techniques.items():
            # Create pipeline
            pipeline = ImbPipeline(steps=[
                ('preprocessor', self.get_preprocessor(self.X)),
                ('oversampler', sampler),
                ('classifier', self.model)
            ])
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                cv=5,
                scoring='average_precision',
                verbose=1,
                n_jobs=-1
            )
            
            # Fit GridSearchCV on the training data
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            self.best_model = best_model

            # Calculate training PR-AUC score
            y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_pred_proba)
            train_f1 = f1_score(y_train, best_model.predict(X_train))

            # Calculate test PR-AUC score
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
            test_f1 = f1_score(y_test, best_model.predict(X_test))

            # Calculate residuals for training and test sets
            residuals_train = y_train - self.best_model.predict(X_train)
            residuals_test = y_test - self.best_model.predict(X_test)

            # Save the best model and parameters
            self.best_model_params = grid_search.best_params_
                
            # Save the best model residuals
            self.best_model_train_residuals[name] = residuals_train
            self.best_model_test_residuals[name] = residuals_test
            
            # Log results
            results.append({
                'model_name': type(self.model).__name__,
                'sampling_technique': name,
                'train_prauc_score': train_pr_auc,
                'test_prauc_score': test_pr_auc,
                'train_f1_score': train_f1,
                'test_f1_score': test_f1
            })

            # Convert results to DataFrame and store in the class attribute
            self.eval_results_df = pd.DataFrame(results)

            # Store feature importances if the model has them
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                feature_importances = best_model.named_steps['classifier'].feature_importances_

                # Optionally, create a DataFrame to log feature importances
                feature_names = self.get_feature_names()
                self.feature_importances_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)

    def get_feature_names(self):
        preprocessor = self.best_model.named_steps['preprocessor']
        # Get feature names from the ColumnTransformer
        numeric_features = preprocessor.transformers_[0][1].get_feature_names_out()
        categorical_features = preprocessor.transformers_[1][1].get_feature_names_out()
        return list(numeric_features) + list(categorical_features)

    def boost_best_model(self, boosting_model, boosting_param_grid, sampling_method):
        # Create list for storing results
        boosting_results = []

        # Use stored residuals
        residuals_train = self.best_model_train_residuals[sampling_method]
        residuals_test = self.best_model_test_residuals[sampling_method]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=self.random_state)

        # Add residuals as a feature
        X_train['residuals'] = residuals_train
        X_test['residuals'] = residuals_test

        # Create pipeline
        boosting_pipeline = ImbPipeline(steps=[
            ('preprocessor', self.get_preprocessor(X_train)),
            ('oversampler', self.sampling_techniques[sampling_method]),
            ('classifier', boosting_model)
        ])

        # Set up GridSearchCV for the boosting model
        grid_search = GridSearchCV(
            estimator=boosting_pipeline,
            param_grid=boosting_param_grid,
            cv=5,
            scoring='average_precision',
            verbose=1,
            n_jobs=-1
        )

        # Fit GridSearchCV on the training data
        grid_search.fit(X_train, y_train)

        # Get the best boosting model
        best_boosting_model = grid_search.best_estimator_

        # Predict on the train set
        y_train_pred_proba = best_boosting_model.predict_proba(X_train)[:, 1]
        y_train_pred = best_boosting_model.predict(X_train)

        # Calculate training evaluation metrics
        train_avg_precision = average_precision_score(y_train, y_train_pred_proba)
        train_f1 = f1_score(y_train, y_train_pred)

        # Predict on the test set
        y_test_pred_proba = best_boosting_model.predict_proba(X_test)[:, 1]
        y_test_pred = best_boosting_model.predict(X_test)

        # Calculate test evaluation metrics
        test_avg_precision = average_precision_score(y_test, y_test_pred_proba)
        test_f1 = f1_score(y_test, y_test_pred)

        # Extract the base model from the pipeline
        base_model = self.best_model.named_steps['classifier']

        # Store and append results
        boosting_result = {
            'base_model': base_model.__class__.__name__,
            'sampling_technique': sampling_method,
            'boosting_model': best_boosting_model.named_steps['classifier'].__class__.__name__,
            'train_prauc_score': train_avg_precision,
            'test_prauc_score': test_avg_precision,
            'train_f1_score': train_f1,
            'test_f1_score': test_f1
        }

        boosting_results.append(boosting_result)

        # Convert boosting results to DataFrame and store in the class attribute
        self.boosting_results_df = pd.DataFrame(boosting_results)