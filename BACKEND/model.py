from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class ToxicityModel:

    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, df):
        X = pd.get_dummies(df.drop(columns=['toxicity']), drop_first=True)
        y = df['toxicity']
        self.model.fit(X, y)
        self.columns = X.columns

    def predict(self, data):
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=self.columns, fill_value=0)

        prob = self.model.predict_proba(df)[0][1]
        pred = self.model.predict(df)[0]

        return prob, pred

    def explain(self):
        return {"info": "Feature importance explanation"}