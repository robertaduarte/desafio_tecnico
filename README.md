# Desafio Técnico

Nesse repositório foi implementado o ESRGAN (_Enhanced Super-Resolution Generative Adversarial Network_), um modelo baseado em redes adversariais generativas que é eficaz e tem uma literatura robusta em tarefas de refinamento e super-resolução de imagens.

### O que é ESRGAN?
ESRGAN é projetada para produzir imagens de super-resolução. Diferente da SRGAN, a ESRGAN introduz blocos residuais que aumentam a estabilidade e profundidade, o uso de uma _loss function_ perceptual que usa _features_ extraídas de redes pré-treinadas (VGG, nesse caso) e um discriminador mais robusto. A ESRGAN é baseada no artigo **[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)** do Wang et al. 2018.


A escolha de uso da ESRGAN nesse desafio é devido aos casos bem sucedidos de aplicação para super resolução em 2x, 4x e 8x. Além disso, o discriminador retorna um valor probabilístico de semelhança entre target e imagem gerada pela geradora. A facilidade de aumentar a ESRGAN para diferentes tarefas como, por exemplo, a geradora retornar _N_ imagens cada qual com diferentes camadas de saída, também é um motivo para utilizar a ESRGAN com esse intuito.

<p align="center">
  <img src="https://github.com/user-attachments/assets/129d6e2a-fa75-4935-a999-93459bc5f4fe" alt="Architecture-of-ESRGAN" width="700"/>
</p>

### Arquitetura

A estrutura principal da ESRGAN, como vista na imagem acima, é composta por três componentes: _generator_, _discriminator_ e uma _perceptual loss_. Abaixo está a descrição detalhada da arquitetura tradicional da ESRGAN:

➡️ ***Generator***

O _generator_ é projetado para aumentar a resolução de imagens de forma realista. Ele é baseado em três partes principais:

1. _First Convolutional Layer_: uma camada convolucional com kernel 3x3 que projeta a imagem de entrada para o _feature space_.

2. _Residual-in-Residual Dense Block (RRDB)_: cada RRDB contém três _Dense Blocks_ conectados com conexões residuais internas e externas: camadas _DenseNet_ que recebem como entrada as camadas anteriores, sem uso de _batch normalization_, função de ativação _LeakyReLU_ e _residual scaling_ com fator de 0.2 para evitar instabilidades.

3. _Upsampling Layers_: uma camada de convolução 3x3 seguida de uma camada de _PixelShuffle_ que dobram a resolução a cada passo. Cada _Upsampling_ é seguido por uma ativação _LeakyReLU_.

4. _Output Convolution_: Uma última convolução 3x3 que projeta o tensor final de volta para o número de canais da imagem RGB (3).

➡️ ***Discriminator*** 

O _discriminator_ da ESRGAN é baseado na _PatchGAN_ que avalia se _patches_ da imagem são reais ou falsos.

1. Conjunto de camadas convolucionais com stride 1 e 2 alternados, seguidos de LeakyReLU.

2. Finaliza com fully connected layers ou uma Adaptive Pooling + 1×1 conv para saída escalar.

➡️ ***Perceptual Loss*** 

A _loss function_ do _generator_ combina: _pixel-wise loss_, _perceptual loss_ (VGG) e _adversarial loss_.

# Meu Modelo
