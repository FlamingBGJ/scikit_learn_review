

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-whitegrid'])# 테머

x = 10 * np.random.rand(50) # 정규분포에서 난수 생성후 10을 곱해줌
y = 2 * x + np.random.rand(50) # x에 2를 곱한다음 + 난수를 더해준다(noise)
# y = 2 * x # 기울기가 2인 직선
plt.scatter(x, y)

# Regression

import numpy as np
import pandas as pd

# 1. linearregression model을 import 
from sklearn.linear_model import LinearRegression

# 2. model 객체 생성, model의 하이퍼 파라미터 선택
model = LinearRegression(fit_intercept=True)

# 학습시킬 dataset의 shape을 확인
x.shape
# x의 shape이 vector 형태가 아니므로 vector 로 변경해준다.
X = x.reshape((50,1))

# X는 x를 vector형태로 변경한 데이터
X.shape

y.shape

# model 학습
model.fit(X, y) # 지도학습이므로 data와 정답을 함께 입력

model.coef_ # 회귀계수, 기울기

model.intercept_ # y 절편의 값

# 학습된 model에 학습시킨 데이터를 다시 넣어 확인
y_predict = model.predict(X)

plt.scatter(x, y_predict, c="r")
plt.scatter(x, y)

import numpy as np

x_test = np.linspace(-1,11, 100) # -1 부터 11까지 데이터 추출
x_test = x_test.reshape((100,1))
y_test = model.predict(x_test)
plt.plot(x_test, y_test, c="r")
plt.scatter(x, y)

# 실습 - 유방암 데이터를 이용한 분류 모델

from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer() # 함수를 실행하여 dataset load

breast_cancer.data #dataset

breast_cancer.target #Label Data(Target Data)

# 총 569개의 데이터, 30개의 특징
# 이미 vector 
print(breast_cancer.data.shape)

# 총 569개의 Label Data
print(breast_cancer.target.shape)

# 30개의 특징에 대한 feature의 정보
print(breast_cancer.feature_names)

import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)

# null 값 확인, object type 확인
data.info()

data.shape

data.head()

data.nunique()

data["target"] = breast_cancer.target

# 0 과 1로 존재
data["target"].nunique()
set(data["target"])

# 데이터의 분포
data["target"].mean() # target 값이 1인 데이터에 대한 분포를 알수있다. 
# target =1 인 데이터가 62% 존재한다.

# 데이터 전처리1
# input data(특성 데이터)와 Label data(정답 데이터)를 분리
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]
print(x_data.shape)
print(y_data.shape)

# model 생성
# LogisticRegression 은 2진 분류 모델이다. 
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()
model_1.fit(x_data, y_data)

# 94% 의 정확도를 보여주고 있다. 
(model_1.predict(x_data) == y_data).mean()

# data를 학습 -> data로 평가 :like 연습문제에서 시험이 다 출제됨
# data 를 분리 한다. 8:2로 분리해서, train data와 valid data로 분리
# x_train : 연습문제
# x_vaild : 모의고사
# 실제 데이터 : 수능시험

# data.iloc[:int(568*0.8), :-1] #train dataset
# data.iloc[int(568*0.8):, :-1] # test dataset

from sklearn.model_selection import train_test_split
x_train, x_vaild, y_train, y_vaild = train_test_split(x_data, y_data, test_size =0.2)

print(x_train.shape)
print(y_train.shape)
print(x_vaild.shape)
print(y_vaild.shape)

model_2 = LogisticRegression()
model_2.fit(x_train, y_train)

(model_2.predict(x_vaild) == y_vaild).mean()
# 90%정확도를 갖고 있다.

from sklearn.metrics import accuracy_score
accuracy_score(y_vaild, model_2.predict(x_vaild))
# accuracy_score 를 이용한 정확도 평가

# 확률로 값을 예측
model_1.predict_proba(x_vaild)[:, 1] #결과 예측 확률
