import torch
import argparse
from PIL import Image
from torchvision import transforms
from model import SGCNet

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='test_images/')
parser.add_argument('--output', default='results/')
args = parser.parse_args()

model = SGCNet()
model.load_state_dict(torch.load("checkpoint.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

with torch.no_grad():
    for name in os.listdir(args.input):
        img = Image.open(os.path.join(args.input, name)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        img_feat = torch.randn(1,512)
        txt_feat = torch.randn(1,512)
        params = model(img_feat, txt_feat)
        enhanced = model.apply_curve(tensor, params)
        enhanced = enhanced.squeeze().permute(1,2,0).numpy()*255
        Image.fromarray(enhanced.astype("uint8")).save(os.path.join(args.output, name))
