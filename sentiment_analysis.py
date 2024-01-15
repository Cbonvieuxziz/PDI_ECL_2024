import nltk

needed_resources = ["names", 
                    "stopwords", 
                    "state_union",
                    "averaged_perceptron_tagger",
                    "punkt"]

nltk.download(needed_resources)

def isDataRelevant():
    """
    Tell whether a data may influence the price of the energy or not
    """
    pass

