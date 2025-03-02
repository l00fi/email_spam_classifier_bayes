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

# Предыдущий реализация наивного байесовского классификатора есть общий случай
# нижебудет реализован наивный байесовский классификатор с логнормальным ядром 
# специально для задачи распознавания спама 

class SpamClassifier:
    def fit(self, data):
        texts = [(data.iloc[i][0].split(), 
                  data.iloc[i][1]) for i in range(data.shape[0])]

        all_cleaned_tokens = []  
        for text in texts:
            all_cleaned_tokens += text[0] # складываю эти слова в один список
    
        unique_tokens = list(set(all_cleaned_tokens)) # с помощью множества избавляюсь от повторений
    
        pre_dataframe = {
            "word": unique_tokens,
            "spam_count": [1]*len(unique_tokens), # 1 для того, чтобы предотвратить 0 при умножении
            "ham_count": [1]*len(unique_tokens) # 1 для того, чтобы предотвратить 0 при умножении
            } 
        
        for text in texts:
            for word in text[0]:
                if word in pre_dataframe["word"]:
                    if "Спам" in text[1]:
                        pre_dataframe["spam_count"][pre_dataframe["word"].index(word)] += 1
                    if "Спам" not in text[1]:
                        pre_dataframe["ham_count"][pre_dataframe["word"].index(word)] += 1
        
        frequency_data = pd.DataFrame(pre_dataframe)

        frequency_data["p_spam"] = np.array(frequency_data["spam_count"].values) * 1/sum(frequency_data["spam_count"].values) # считаю вероятнось появление каждого отдельного слова
        frequency_data["p_ham"] = np.array(frequency_data["ham_count"].values) * 1/sum(frequency_data["ham_count"].values) # считаю вероятнось появление каждого отдельного слова
        probs = np.array(data["sign"].value_counts())/data.shape[0] # вероятность каждого класса

        self.class_proba = {
            "spam": probs[1],
            "ham": probs[0] 
            } # запоминаем вероятности каждого класса
        self.tokens_prob = frequency_data # запоминаем вероятность каждого слова

        return self
    
    def predict(self, text):
        this_tokens_rows = self.tokens_prob.loc[self.tokens_prob['word'].isin(text)]

        p_spam = np.log(self.class_proba["spam"]) + sum([np.log(p) for p in this_tokens_rows["p_spam"].values])
        p_ham = np.log(self.class_proba["ham"]) + sum([np.log(p) for p in this_tokens_rows["p_ham"].values])

        if p_spam > p_ham:
            return "Спам"
        else:
            return "Входящие"

 