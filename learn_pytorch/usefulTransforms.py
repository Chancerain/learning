from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)

print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_toTensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

writer.close()