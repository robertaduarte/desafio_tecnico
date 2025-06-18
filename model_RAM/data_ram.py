#Image Enhancer
#by Roberta Duarte


from data import createDataset
from torch.utils.data import DataLoader


lr_dir = "C:\\Users\\impor\\Downloads\\prova_tecnica\\lr_low\\"
hr_dir = "C:\\Users\\impor\\Downloads\\prova_tecnica\\lr_mild\\"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


dataset = createDataset(lr_dir, hr_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for i, (lr, hr) in enumerate(dataloader):
    lr, hr = lr.to(device), hr.to(device)
