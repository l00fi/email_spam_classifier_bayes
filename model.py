import pandas as pd
import numpy as np
import scipy.stats as st

# Математическое описание реализованного алгоритма по следующей ссылке: 
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Training

class NormalNaiveBayes:
    def fit(self, X, y):
        dataset = np.column_stack((X, y))
        data_per_classes = {}
        for row in dataset:
            class_k = row[-1]
            if class_k in data_per_classes.keys():
                data_per_classes[class_k].append(row[0:-1])
            else:
                data_per_classes[class_k] = list([row[0:-1]])

        self.p_classes = {class_data:len(data_per_classes[class_data])/len(X) for class_data in data_per_classes.keys()}

        features_distribs = {}
        for class_k in data_per_classes.keys():
            data = pd.DataFrame(data_per_classes[class_k])
            var = data.std(ddof=1)
            mean = data.mean()
            var_mean_dataframe = pd.DataFrame([var, mean])
            features_distribs[class_k] = dict()
            for column in var_mean_dataframe.columns:
                params = var_mean_dataframe[column].tolist()
                features_distribs[class_k][column] = st.norm(params[1], params[0])

        self.features_distribs = features_distribs
        return self
    
    def predict_prob(self, X):
        prob_per_class_for_x = list()
        for x in X:
            evidence = 0
            probs = list()
            for class_k in self.features_distribs:
                posterior_denumerator_k = self.p_classes[class_k]
                distribs = self.features_distribs[class_k]
                for distrubs_key in distribs.keys():
                    posterior_denumerator_k *= distribs[distrubs_key].pdf(x[distrubs_key])
                probs.append(posterior_denumerator_k)
                evidence += posterior_denumerator_k
            prob_per_class_for_x.append(np.array(probs) / evidence)
        return prob_per_class_for_x
    
    def predict(self, X):
        probs = self.predict_prob(X)
        predict = list()
        class_labels = list(self.p_classes.keys())
        for prob in probs:
            predict.append(class_labels[prob.argmax()])
        return np.array(predict)