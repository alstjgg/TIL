# 01.24 - Tensorflow
## Tensorflow
- library for implementing machine learning that requires a large number of mathematical operations
- 여러 기계에서 평행적인 계산을 실행하여 빠른 연산을 가능하게 해줌.

## Tensor
![image](https://cdn-images-1.medium.com/max/1600/1*jAzi88LbxU3q0-5iJ1cjyw.png)
- a geometric object that maps in a multi-linear manner geometirc vectors, scalars, and other tensors to a resulting tensor.
- a multidimensional array / N-dimensional vector
    -> represent N-dimensional datasets
    -> flow between networks
- constant / variable

## Computational graph
![image](https://www.tensorflow.org/images/tensors_flowing.gif)
- Node: operations / Edge: tensor
    -> graph를 선언하고 session을 통해서 실행하면, tensor가 operation에 입력되고 다른 tensor가 출력되어 flow를 따라 연산
    -> 최종 output을 통해 가중치를 업데이트하는데, optimizer의 w1, b1, w2, b2 등의 가중치들을 업데이트하면 하나의 session이 끝남
        - w: weight / b: bias

```python
sess = tf.Session()
sess.run(task)
sess.close()
```
## 사용 함수
- 웬만한 수학 연산은 모두 구현되어 있다 -> [Tensoflow API](https://www.tensorflow.org/api_docs/python/)
### 1. tf.placeholder()
```python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```
- 결과를 예측하기 위해 필요한 feature의 갯수와 제공되는 데이터의 개수가 input 데이터의 shape를 결정한다
    -> 데이터의 개수는 변동이 있을 수 있으므로 [?, feature수] 형태의 shape을 가지는데, 이는 `[None, n]`으로 표현
    -> Feature(X) / Label(y)
- `x_data = np.array([[..], [..], ..])` -> feature들의 값 저장
- `y_data = np.array([[..], [..], ..])` -> label의 값 저장

### 2. tf.Variable()
```python
tf.Variable(<initial-value>, name=<optional-name>)
```
- weight와 bias와 같은 값을 선언
- 보통 initial value는 랜덤하게 초기화를 시키기 때문에, X와 y의 shape만을 고려한 채로 `tf.random_normal()` 사용

### 3. tf.matmul()
- 행렬곱

### 4. tf.train module
- [Training modules](https://www.tensorflow.org/api_docs/python/tf/train)
-> optimizer 선택

## Linear Regression example
```python
import tensorflow as tf
```
- import tensorflow library
```python
x_data = np.array([[3, 1], [4, 0], [5, 1]])
y_data = np.array([[120000], [100000], [200000]])
```
- feature는 2개, label은 1개인 데이터 3개
```python
# hyper parameter
lr = 0.01
n_epoch = 2000
```
- learning rate `lr`과 epoch `n_epoch` 정의
```python
X = tf.placeholder(tf.float32, shape=[None, 2], name="X")
y = tf.placeholder(tf.float32, shape=[None, 1], name="y")

W = tf.Variable(tf.random_normal([2, 1]), name='wight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```
- input 데이터 `X`와 output 데이터 `y` 정의 -> 위의 `x_data`와 `y_data`의 shape과 각각 동일하게 설정
- weight `W`와 bias `b` 초기화 -> 랜덤하게 초기화하며 data shape 고려
```python
hypothesis = tf.matmul(X, W) + b
```
- linear regression 모델의 hypothesis 정의
```python
cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
```
- linear regression의 `cost`는 모든 데이터들의 예측값(`hypothesis`)와 실제값(`y`)의 차이(의 제곱)의 평균
- `train` 단계는 gradient descent 방법을 차용하여 `cost`를 최소화하도록 variable 재설정
```python
with tf.Session() as sess:
    # 변수가 있는 경우에는 초기화를 실행해줘야 한다.
    sess.run(tf.global_variables_initializer())
    # train이 반환하는 값은 우리에게 필요없다.
    for step in range(n_epoch):
        c, _ = sess.run([cost, train], feed_dict={X: x_data, y: y_data})
        if step % 500 == 0:
            print("Step :", step, "Cost :", c)
            # x, y를 임의로 만든거라..
            # 이 부분은 train data를 학습시키는지 확인하는 목적 외에는 없다.
            print(sess.run(hypothesis, feed_dict={X: x_data}))
```
- run session

## Tensorflow 설치
- Tensorflow cpu 버젼은 현재 windows에서 python 3.6만 호환되기 때문에 3.7을 사용하고 있다면 3.6 interpreter를 사용해야 한다.
