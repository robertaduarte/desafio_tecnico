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

3. _Upsampling Layers_: uma camada de _PixelShuffle_ que dobram a resolução a cada passo seguida por uma ativação _LeakyReLU_.

4. _Output Convolution_: Uma última convolução 3x3 que projeta o tensor final de volta para o número de canais da imagem RGB (3).

➡️ ***Discriminator*** 

O _discriminator_ da ESRGAN é baseado na _PatchGAN_ que avalia se _patches_ da imagem são reais ou falsos.

1. Conjunto de camadas convolucionais com stride 1 e 2 alternados, seguidos de LeakyReLU.

2. Finaliza com fully connected layers ou uma Adaptive Pooling + 1×1 conv para saída escalar.

➡️ ***Perceptual Loss*** 

A _loss function_ do _generator_ combina: _pixel-wise loss_, _perceptual loss_ (VGG) e _adversarial loss_.

# Meu Modelo

O meu modelo segue o formato de uma ESRGAN tradicional com algumas alterações específicas para o problema. Abaixo, descreverei as mudanças que foram feitas em cada arquivo e os problemas que resultaram nessas mudanças:

`architecture.py`: a camada de arquitetura foi implementada seguindo os passos do artigo  **[Wang et al. 2018](https://arxiv.org/abs/1809.00219)**. 

1. A primeira mudança está nas últimas camadas da geradora que foram alteradas para retornar 3 imagens em vez de 1 só. Cada imagem é resultado de uma camada de saída diferente, mais especificamente:
    - Imagem 1: a camada final é um bloco onde tem duas vezes uma convolução 9x9 seguida de uma _LeakyReLU_ e uma convolução final 9x9.
    - Imagem 2: a camada final é uma convolução 9x9 seguida de uma _LeakyReLU_ e uma convolução final 9x9.
    - Imagem 3: a camada final é uma convolução 9x9.
      
2. A segunda mudança é no adicional de um condicional de escalas porque uma escala de 2 precisa de apenas 1 _Upsampling Layer_ enquanto uma escala de 8 precisa de 3 _Upsampling Layers_. Cada _Upsampling Layer_ aumenta a imagem em 2x. Então dependendo da condicional que é recebida na classe da _generator_, o modelo se organiza para ser treinado com a escala escolhida.

3. Adicionei uma convolução 3x3 antes da camada _PixelShuffle_ para refinar o treinamento.

`perceptual_loss.py`: implementa a _perceptual loss_ baseada no modelo pré-treinado VGG para comparar as features.

`data.py`: o arquivo possui funções que tratam os dados e preparam batches, um de cada vez. Apesar da preparação em loop, esse formato é eficiente para não gastar a RAM quando todos dados são guarados de uma vez em um _dataloader_ (na pasta `modelo_RAM` tem um `data.py` que cria um _dataloader_ que pode ser eficiente para dados menores).

`train.py`: o arquivo possui o loop de treinamento que considera batches separados. Além disso, primeiro a discriminadora é treinada e depois a geradora é treinada onde ela devolve 3 imagens que são comparadas com o target e a _loss function_ é somada. Ao final, o modelo chama o arquivo `evaluation.py` que plota os resultados ao final de cada época assim como as métricas de validação.

`evaluation.py`: arquivo realiza a avaliação em cima das imagens de validação e retorna a imagem gerada no diretório, a imagem possui as 3 imagens geradas pela _generator_ assim como as métricas MSE, SSIM e PNSR para cada uma em comparaão ao target.

`score.py`: 

`requirements.txt`: todos requisitos para rodar o código

### Hiperparâmetros

Nesse trabalho, os hiperparâmetros foram:

1. Número de camadas de convolução na saída da geradora para cada uma das 3 imagens
2. Número de filtros
3. _Batch size_

Como a ESRGAN tem uma literatura robusta e bem estabelecida, diversos hiperparâmetros tradicionais (como _learning rate_, pesos das loss, etc) já eram estabelecidos como _default_ para esse caso.

### Métricas

As métricas utilizadas para avaliar o resultado na parte de validação são métricas usadas tradicionalmente em análises de problemas de visão computacional:

1.  MSE (Mean Squared Error): avalia, em valor absoluto, a diferença entre a imagem gerada e o _target_ como a média dos _pixels_.
2.  PSNR (Peak Signal-to-Noise Ratio): representa a qualidade da imagem gerada em relação ao _target_.
3.  SSIM (Structural Similarity Index Measure): avalia se as imagens têm a mesma estrutura e aparência visual.


# Resultados

## Sandbox

Na pasta,  `modelo_RAM` há versões dos arquivos `data.py`, `train.py` e `evaluation.py` para uso com _dataloader_.

