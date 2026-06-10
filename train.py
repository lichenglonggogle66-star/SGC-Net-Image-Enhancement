import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SGCNet
from loss import JointLoss
from dataset import MIT5K

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./mit5k')
parser.add_argument('--batch', default=8)
parser.add_argument('--epochs', default=100)
parser.add_argument('--lr', default=1e-4)
args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
dataset = MIT5K(args.dataset, transform)
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

model = SGCNet().cuda()
criterion = JointLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for img, gt, _ in loader:
        img, gt = img.cuda(), gt.cuda()
        img_feat = torch.randn(img.shape[0],512).cuda()
        txt_feat = torch.randn(img.shape[0],512).cuda()
        params = model(img_feat, txt_feat)
        out = model.apply_curve(img, params)
        loss = criterion(out, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "checkpoint.pth")
