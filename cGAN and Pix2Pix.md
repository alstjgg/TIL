# cGAN (Conditional GAN)
## 개요
[Conditional Gernerative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf) by Mehdi Mirza, Simon Osindero
- GAN을 사용하면 학습에 사용한 이미지와 비슷한 이미지를 생성해낼 수 잇는데, 어떤 이미지를 생성할지는 선택할 수 없음
- cGAN의 경우 어떤 이미지를 생성할지 제어할 수 있음 -> Pix2Pix, CycleGAN
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/1.PNG)
- Generator가 random noise를 input으로 받아 학습 이미지와 비슷한 이미지를 생성하는 과정
- Latent Variable: 직접적으로 관측되는 변수가 아닌, 관측이 가능한 다른 변수들로부터 추론이 가능한 변수
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/2.PNG)
- Autoencoder에서 input data를 압축하여 이를 latent variable z로 사용한다
- 뒷부분은 Generator의 과정과 유사
-> latent variable 자체에 이미지 생성과 관련된 condition을 부여하여 이를 통해 생성되는 이미지를 제아할 수 있다 -> cGAN
## 학습 방법
- condition 주기: 특정 조건을 나타내는 정보 y를 가해준다
    - ex. class label 추가 -> 특정 class의 결과만 생성
- Objective Function
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/3.PNG)

-> y가 조건부로 들어감

# Pix2Pix (image-to-image translation)
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) by Phillip Isola,  Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/4.PNG)
- 영상 변환 및 생성을 위한 범용 framework
- cGAN의 일종
- cGAN의 한계: y는 one-hot encoding 방식을 사용한 크지 않은 벡터이기 때문에, 소수의 class 이미지를 생성하는 것은 가능하지만 복잡한 class나 조건이 많은 이미지 생성은 어렵다
    - ex. edge-map에서 특정 대상 생성 / 흑백에서 컬러로 전환
## 장점
- 0일반적인 CNN 보다 학습에 필요한 데이터가 적다
- 데이터 쌍 (x, y) (input, 목표 이미지)을 제공하면, 변환의 종류와 상관없이 이미지 생성이 가능하다
- 기존 알고리즘들의 loss function과 다름
    - 기존
        - L1 또는 L2 loss를 최소화시키는 방향 -> 이미지들의 평균을 취하여 영상이 선명하지 않음
        - pixel-to-pixel 변환에 초점 (인접한 pixel 끼리 서로 영향을 끼치지 않고 독립적이라는 가정: unstructured loss)
    - Pix2Pix
        - GAN에 기반 -> 선명한 이미지 생성
        - pixel과 pixel간의 joint configuration를 고려한 structured loss 개념 적용
## Objective Function
- Inpainting: 이미지 위의 얇은 선폭을 가진 낙서나 자막을 제거 / 넓은 영역이 비어있는 이미지를 채우기
    -> 단순한 영상 복원을 위한 reconstruction loss(L1, L2 loss)가 아닌, adversarial loss를 결합하여 더 선명한 영상 생성 가능
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/5.PNG)![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/6.PNG)
- Generator는 latent vector z(random noise)와 입력 영상(x)이 인가되어 이미지를 생성한다
    - latent variable z가 달라지면 다른 이미지를 생성함(z 없이는 deterministic한 generator가 됨)
    - z는 생성되는 이미지의 condition으로 작용
- Discriminator에게 분류해야할 데이터(G(x, z) 또는 y) 외에 입력 영상(x)가 함께 인가된다
- Adversarial loss(진짜 이미지처럼 보이게 해줌) + L1 Reconstruction loss(ground truth x와 비슷한 이미지가 생성되도록 해줌) 이용
- L1 loss

    ![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/7.PNG)
- Objective Function

    ![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/8.PNG)
## U-NET
- Generator에서 사용하는 구조
- encoder-decoder 구조에서 영상크기를 축소/확대 시키는 과정에서 영상의 선명함을 잃는 것을 피하기 위해 skip connection을 사용한다
- FCN(Fully Convolutional Network)의 경우 1/8영상부터 skip connection을 사용하지만 U-Net의 경우 첫번째 convolution layer부터 적용한다
- encoder에서 decoder로 정보를 직접 넘기기 때문에 선명한 이미지가 생성된다
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/9.PNG)

-> 회색화살표: skip-connection
## PatchGAN
- Discriminator에서 사용하는 구조

![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/10.PNG)
- Generator의 objective function인 L1 reconstruction loss는 이미지 간의 유클리드 거리를 최소화하는 방향으로 접근하기 때문의 이미지의 평균 성분인 저주파에 집중하게 된다
    -> 따라서 Discriminator는 고주파 영역에 집중하여 이미지의 true/false를 구별한다
    
![image.png](https://raw.githubusercontent.com/alstjgg/alstjgg.github.io/master/cGAN_and_Pix2Pix/11.PNG)
- ImageGAN: 일반적으로 GAN에서는 이미지 전체의 score를 구한다
- PatchGAN: 전체 영역이 아닌, 이미지를 특정 크기의 patch로 분리하여 단위별로 true/false를 구한 후, 그 결과의 평균을 구한다
    -> pixel 간의 correlation은 거리에 반비례함 (일정 거리 이상 멀어지면 연관성이 없어짐)
    -> true인 patch가 많아지는 방향으로 학습
- PixelGAN: patch의 크기를 pixel 단계까지 줄인 1x1 PatchGAN
-> 이미지에서 correlation 범위를 적절하게 선택하여 patch의 크기를 정해야 함 (전체 이미지 크기 및 영상 전체에서 특정 픽셀과 다른 픽셀들 간의 연관 관계가 미치는 범위 고려)
    -> Patch 크기는 hyper-parameter로 취급

# 결론
1. Generator는 encoder-decoder 형태를 취한다
2. 최적화를 위해 cGAN loss와 L1 reconstruction loss를 고려하여 Objective Function을 설계한다
3. Generator는 선명한 화질을 위해 U-Net 구조를 취한다
4. Discriminator는 빠른 연산과 구조적 유연함을 위해 PatchGAN 구조를 취한다
