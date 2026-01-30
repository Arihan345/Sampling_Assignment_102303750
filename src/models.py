from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_models():
    return {
        "M1": LogisticRegression(max_iter=5000,solver="lbfgs"),
        "M2": DecisionTreeClassifier(),
        "M3": RandomForestClassifier(),
        "M4": KNeighborsClassifier(),
        "M5": GaussianNB()
    }
