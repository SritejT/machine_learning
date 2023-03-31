import os
import email.parser
import email.policy
from sklearn.pipeline import Pipeline
from CustomFiles.EmailTransformer import EmailTransformer, VectorTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

filenames_ham = [name for name in sorted(os.listdir("datasets/spam_classifier/easy_ham"))]

filenames_spam = [name for name in sorted(os.listdir("datasets/spam_classifier/spam"))]


def load_email(is_spam, filename, spam_path="datasets/spam_classifier"):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_data = [load_email(is_spam=False, filename=name) for name in filenames_ham]

spam_data = [load_email(is_spam=True, filename=name) for name in filenames_spam]

all_data = ham_data + spam_data

test_data = (["ham"] * len(ham_data)) + (["spam"] * len(spam_data))

X_train, X_test, y_train, y_test = train_test_split(all_data, test_data, test_size=0.2)

"""
------------------------------------------------------------------------------------------------------------------------
"""

full_pipeline = Pipeline([
    ("email", EmailTransformer()),
    ("vector", VectorTransformer()),
])

X_train_prepared = full_pipeline.fit_transform(X_train)

forest_clf = RandomForestClassifier()

forest_clf.fit(X_train_prepared, y_train)

print("Trained Model!")

X_test_prepared = full_pipeline.fit_transform(X_test)

print(cross_val_score(forest_clf, X_test_prepared, y_test, scoring="accuracy", cv=3).mean())
