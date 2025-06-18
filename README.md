# Desafio Técnico

Nesse repositório foi implementado o ESRGAN (_Enhanced Super-Resolution Generative Adversarial Network_), um modelo baseado em redes adversariais generativas que é eficaz e tem uma literatura robusta em tarefas de refinamento e super-resolução de imagens.

### O que é ESRGAN?
ESRGAN é projetada para produzir imagens de super-resolução. Diferente da SRGAN, a ESRGAN introduz blocos residuais que aumentam a estabilidade e profundidade, o uso de uma _loss function_ perceptual que usa _features_ extraídas de redes pré-treinadas (VGG, nesse caso) e um discriminador mais robusto. 

A escolha de uso da ESRGAN nesse desafio é devido aos casos bem sucedidos de aplicação para super resolução em 2x, 4x e 8x. Além disso, o discriminador retorna um valor probabilístico de semelhança entre target e imagem gerada pela geradora. A facilidade de aumentar a ESRGAN para diferentes tarefas como, por exemplo, a geradora retornar _N_ imagens cada qual com diferentes camadas de saída, também é um motivo para utilizar a ESRGAN com esse intuito.
