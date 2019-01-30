# GAN (Generative Adversarial Networks)
## Discriminative Network vs. Generative Adversarial Network
### Discriminative Network
- Classification / Segmentation -> 대상 판별
- input data x 에 대해 output data(label)이 y가 될 확률 도출 -> P(y|X)
- cost function / loss function을 정의하고, 오차를 back-propagate하여 변수를 업데이트 -> 한개의 network 학습
### Generative Adversarial Network
- 특정 확률 분포 P_data(X)를 학습 -> 유사한 분포 P_model(x)를 갖는 데이터를 도출
- label 없이 데이터들의 representation(중요한 정보)를 찾아내어 비슷한 정보를 가진 데이터셋 생성
- Generator + Discriminator -> 경쟁적으로 학습 시켜 진짜와 가짜를 구별하기 힘들때까지 발전시킴 -> 2개의 network 학습
1. Generator 고정, Discriminator 학습
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/1.png)
2. Discriminator 고정, Generator 학습
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/2.png)

-> 2 player mini-max 게임과 비슷

## GAN 개요
### 생성모델의 분류
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/3.PNG)
- 많은 모델들은 Maximum Likelihood에 기반을 두고 있다
- Explicit density: 모델에 대한 확률 변수를 직접접으로 구함
- Implicit density: 모델에 대한 확률 변수를 구하지 않음 -> uses a likelihood function that cannot be expressed explicitly. the form of the likelihood function or any derive quantaties are not required, but maximizing likelihood under certain coniditons can be shown.
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/4.PNG)
- P_data(X)와 P_model(X)의 차가 최소가 되도록 한다

### Autoencoder
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/7.PNG)
- Input random noise `z` -> Generator network -> Sample
- Random noise를 입력 받아 디코더를 잘 학습시키면 원하는 분포를 갖는 데이터를 생성할 수 있다
### Objective Function
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/5.PNG)
- 첫번째 항: D가 학습 데이터를 진짜(1)라고 판단할 수 있는 능력
- 두번째 항: D가 G에서 만든 데이터(=G(z))를 가짜(0)라고 판정할 수 있는 능력
-> D: 최대화 / G: 최소화(두번째 항 최소화)
- G 학습의 문제점: `1-D(G(z))`를 최소화시키고자 하면 기울기(Gradient)가 너무 작아 학습이 어려움
    -> `D(G(z))`를 최대화시키는 방향으로 학습 -> 기울기(Gradient) 확보 가능
        -> *log 함수 그려보면 확인 가능*

![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/GAN/6.PNG)
- 학습 종료: 사람 개입 *-> 차이가 특정 수치 이하면 종료(이전 논문의 방식)*
- k=1이면 G와 D를 번갈아 가며 학습하는 형식

