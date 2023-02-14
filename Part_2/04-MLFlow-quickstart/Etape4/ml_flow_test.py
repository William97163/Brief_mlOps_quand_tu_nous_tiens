from TaxiFareModel.data import get_data
from sklearn.linear_model import LinearRegression
from TaxiFareModel.trainer import Trainer
from sklearn.model_selection import train_test_split

data = get_data("../data_new_york/train.csv")

y = data.pop("fare_amount")
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trainer = Trainer(LinearRegression())

pipe = trainer.set_pipeline()

search = trainer.run(X_train, y_train, pipe)

print(trainer.evaluate(X_test, y_test, search))

