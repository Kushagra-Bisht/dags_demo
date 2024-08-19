import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')
iris = load_iris()
X = iris.data  # Corrected: removed parentheses
y = iris.target  # Corrected: removed parentheses

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 10


mlflow.set_experiment('iris.dt2')
with mlflow.start_run(run_name='Campusx'):
    rf=DecisionTreeClassifier(max_depth=max_depth)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)

    cf=confusion_matrix(y_pred,y_test)
    sns.heatmap(cf,annot=True,cmap='Blues')
    plt.title('Confusion_Matrix')
    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf,"Decision Tree")
    mlflow.set_tag('author',"Kushagra Bisht")

    
    print(accuracy)
