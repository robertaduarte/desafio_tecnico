import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from architecture import *
from data import proximo_batch

lr_dir_val = "/diretorio/das/imagens/de/validacao/de/baixa/resolucao/"
hr_dir_val =  "/diretorio/das/imagens/de/validacao/de/alta/resolucao/"

scale = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

generator = Generator(scale).to(device)
generator.load_state_dict(torch.load("/diretorio/dos/pesos/da/geradora"))
generator.eval()

def score(generator, lr, hr):
    """
    Args:
        generator: a geradora treinada
        lr: imagem de baixa resolução 
        hr: imagem de alta resolução 

    Returns:
        dicionário com métricas e ranking das imagens da melhor pra pior
    """

    weights = {'psnr': 0.4, 'ssim': 0.4, 'mse': 0.2}  # pesos ajustáveis

    with torch.no_grad():
        fake_hr1, fake_hr2, fake_hr3 = generator(lr)

    outputs = [fake_hr1, fake_hr2, fake_hr3]
    metric_results = []

    for i, fake in enumerate(outputs):
        mse_val = F.mse_loss(fake, hr).item()
        psnr_val = psnr(hr.squeeze().permute(2, 1, 0).cpu().numpy(), fake.squeeze().permute(2, 1, 0).cpu().numpy(), data_range=1.0)
        ssim_val = ssim(hr.squeeze().permute(2, 1, 0).cpu().numpy(), fake.squeeze().permute(2, 1, 0).cpu().numpy(), data_range=1.0, channel_axis=-1)

        '''
        Ajustando o score com os pesos
        '''
        score = (
            weights['psnr'] * psnr_val +
            weights['ssim'] * ssim_val -
            weights['mse'] * mse_val 
        )

        metric_results.append({
            'index': i + 1,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'mse': mse_val,
            'score': score
        })

    # Ordena pelo score
    ranked = sorted(metric_results, key=lambda x: x['score'], reverse=True)

    '''
    Probabilidade de confusão usando o softmax 
    '''
    scores = np.array([r['score'] for r in ranked])
    probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
    for i, p in enumerate(probs):
        ranked[i]['prob'] = p

    '''
    mostra o ranking das imagens, com as mais parecida primeiro
    '''
    for r in ranked:
        print(f"Fake_HR{r['index']}: PSNR={r['psnr']:.2f}, SSIM={r['ssim']:.4f}, MSE={r['mse']:.6f}, Score={r['score']:.3f}, Probabilidade de confusão: {r['prob']:.2%}")

    return ranked

scale = 4
batch_size = 1

if scale == 2:

  img_size_input = (256, 128)
  img_size_target = (512, 256)

elif scale == 4:

  img_size_input = (128, 64)
  img_size_target = (512, 256)

else: 

  img_size_input = (128, 64)
  img_size_target = (1024, 512)

index = 0

for i in range(100-batch_size):
    lr, _ = proximo_batch(lr_dir_val, index, img_size_input, batch_size)
    hr, index = proximo_batch(hr_dir_val, index, img_size_target, batch_size)
    score(generator, lr, hr)

