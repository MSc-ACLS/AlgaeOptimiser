import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
from preprocessing import preprocess_data

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R2 Score: {r2_score(y_test, y_pred)}')
    # Feature importances
    importances = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df)
    
def main():
    zhaw_data, agroscope_data = preprocess_data()
    X = zhaw_data.drop(['Dryweight'], axis=1)
    y = zhaw_data['Dryweight']
    #X = agroscope_data.drop(['Dryweight'], axis=1)
    #y = agroscope_data['Dryweight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__": main()
