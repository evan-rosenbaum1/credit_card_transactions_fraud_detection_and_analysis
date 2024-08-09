import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelEvaluator:
    def __init__(self, model, param_grid, sampling_techniques, random_state=19):
        self.model = model
        self.param_grid = param_grid
        self.sampling_techniques = sampling_techniques
        self.random_state = random_state
        self.results = []

    def evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        
        for name, sampler in self.sampling_techniques.items():
            # Create pipeline
            pipeline = ImbPipeline(steps=[
                ('preprocessor', self._get_preprocessor(X)),
                ('oversampler', sampler),
                ('classifier', self.model)
            ])
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grid,
                cv=5,
                scoring='average_precision',
                verbose=2,
                n_jobs=-1
            )
            
            # Fit GridSearchCV on the training data
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_

            # Calculate training PR-AUC score
            y_train_pred_proba = best_model.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_pred_proba)
            train_f1 = f1_score(y_train, best_model.predict(X_train))

            # Calculate test PR-AUC score
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
            test_f1 = f1_score(y_test, best_model.predict(X_test))

            # Log results
            self.results.append({
                'model_name': type(self.model).__name__,
                'sampling_technique': name,
                'train_prauc_score': train_pr_auc,
                'test_prauc_score': test_pr_auc,
                'train_f1_score': train_f1,
                'test_f1_score': test_f1
            })

        return self.results

    def _get_preprocessor(self, X):
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])