import cv2
import torch
import torchvision.transforms as transforms
from models.enhancer import RetinexNet
from piq import LPIPS, psnr, ssim
from torchvision.utils import make_grid, save_image


def preprocess(img):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([512, 512], antialias=True)]
    )
    return tfm(img).unsqueeze(0).clamp(0, 1)


def iq_compare(a, b):
    metrics =  {
        "psnr": psnr(a, b),
        "ssim": ssim(a, b),
        "lpips": LPIPS()(a, b),
    }
    for k, v in metrics.items():
        metrics[k] = v.detach().cpu()
    return metrics



if __name__ == "__main__":
    frame_no = 391
    night = preprocess(
        cv2.imread(
            f"/data/matthew/dark_zurich/val/rgb_anon/val/night/GOPR0356/GOPR0356_frame_{frame_no:06}_rgb_anon.png"
        )
    )
    day = preprocess(
        cv2.imread(
            f"/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref/GOPR0356_frame_{frame_no:06}_ref_rgb_anon.png"
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = RetinexNet().to(device)
    net.load_state_dict(torch.load("decomp.pth"))
    net.eval()

    r1, i1 = net(night.to(device))
    r2, i2 = net(day.to(device))
    save_image(make_grid(torch.concatenate([night, day])), 'out.png')
    save_image(make_grid(torch.concatenate([r1, r2])), 'out_r.png')
    save_image(make_grid(torch.concatenate([i1, i2])), 'out_i.png')
    print('RGB images')
    print(iq_compare(day, night))
    print('Retinex Reflectance')
    print(iq_compare(r1, r2))
    print('Retinex Illumination')
    print(iq_compare(i1, i2))
