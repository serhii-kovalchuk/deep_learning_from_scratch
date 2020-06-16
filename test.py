from sklearn.datasets import load_digits
from classifiers import Network
from sklearn.ensemble import BaggingClassifier


def get_training_and_test_sets(digits):
    return (
        digits['data'][:1400], 
        digits['target'][:1400], 
        digits['data'][1400:], 
        digits['target'][1400:]
    )

digits = load_digits()
x_train, y_train, x_test, y_test = get_training_and_test_sets(digits)


network_clf = Network([64, 50, 10], activation="sigmoid", alpha=0.16)
network_clf.fit(x_train, y_train, etha=3, n_epoches=7000, minibatch_size=400)

print("network clasifier train score:", network_clf.score(x_train, y_train))
print("network clasifier test score:", network_clf.score(x_test, y_test))

network_clf = Network([64, 50, 10], activation="sigmoid", alpha=0.16)
bagging_clf = BaggingClassifier(
    base_estimator=network_clf,
    n_estimators=25,
    max_samples=400,
    bootstrap=True,
    n_jobs=-1
).fit(x_train, y_train)

print("bagging network clasifier train score:", bagging_clf.score(x_train, y_train))
print("bagging network clasifier test score:", bagging_clf.score(x_test, y_test))


