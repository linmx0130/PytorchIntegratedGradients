import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
noNormTransform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
toPIL = T.ToPILImage()

def readAndCropImage(filename, showImg=False, showCropImg=False):
    im = Image.open(filename)
    if showImg:
        im.show()
    imt = transform(im)
    nimt = noNormTransform(im)
    if showCropImg:
        nim = toPIL(nimt)
        nim.show()
    return imt, nimt
    
def buildGradImage(gimt, nimt, thresh = 0.05):
    gimt -= gimt.min()
    gimt /= gimt.max()
    gimt = gimt.mean(dim=0)
    sortedGradView = gimt.flatten().sort().values
    upThresh = sortedGradView[int(len(sortedGradView) * (1.0 - thresh / 2))]
    downThresh = sortedGradView[int(len(sortedGradView) * (thresh/2))]
    imgGradView = ((gimt > upThresh) + (gimt < downThresh))
    imgGradView = imgGradView.resize(1, 224, 224)
    imgGradView = nimt * torch.cat([imgGradView, imgGradView, imgGradView])
    imgGradView = toPIL(imgGradView)
    return imgGradView

def buildFastGradAttackImage(filename, theta=0.07):
    rebuildTransform = T.Compose([T.Resize(256), T.ToTensor()])
    baseImg = Image.open(filename)
    baseImgT = rebuildTransform(baseImg)
    c, H, W = baseImgT.shape
    imt, nimt = readAndCropImage(filename, showImg=False, showCropImg=False)
    imt = imt.resize(1, 3, 224, 224)
    gimt = imt.clone().detach().requires_grad_(True)
    cls = resnet18(gimt)
    target = torch.tensor([cls.argmax()])
    loss = torch.nn.functional.cross_entropy(cls, target)
    loss.backward()
    attack_imt = nimt + theta * torch.sign(gimt.grad[0])
    attack_imt = torch.min(attack_imt, torch.ones_like(attack_imt))
    attack_imt = torch.max(attack_imt, torch.zeros_like(attack_imt))

    offsetH = H // 2 - 112
    offsetW = W // 2 - 112
    baseImgT[:, offsetH: offsetH + 224, offsetW : offsetW + 224] = attack_imt
    attack_im = toPIL(baseImgT)
    return attack_im

def gradientMethod(filename):
    imt, nimt = readAndCropImage(filename, showImg=False, showCropImg=False)
    imt = imt.resize(1, 3, 224, 224)
    gimt = imt.clone().detach().requires_grad_(True)
    cls = resnet18(gimt)
    oneHotGrad = torch.zeros_like(cls)
    oneHotGrad[0][cls.argmax()] = 1
    cls.backward(oneHotGrad)
    imgGradView = buildGradImage(gimt.grad[0], nimt)
    croppedImg = toPIL(nimt)
    return imgGradView, croppedImg, int(cls.argmax())

def integratedGradient(filename, baseline=None, sampleSize=10, thresh=0.05):
    imt, nimt = readAndCropImage(filename, showImg=False, showCropImg=False)
    if baseline is None:
        baseline = torch.zeros_like(imt)
    alpha_range = torch.arange(1.0 / sampleSize, 1.0 + 1.0/sampleSize, 1.0 / sampleSize)

    batch_input = torch.zeros([len(alpha_range), 3, 224, 224])
    difference = imt - baseline
    for idx, alpha in enumerate(alpha_range):
        batch_input[idx] = baseline + alpha * difference
    batch_input = batch_input.clone().detach().requires_grad_(True)

    cls = resnet18(batch_input)
    clsResult = cls[-1].argmax()
    clsScores = torch.nn.functional.softmax(cls[-1])
    clsScore = clsScores[clsResult]
    #oneHotGrad = torch.zeros_like(cls)
    #print(oneHotGrad.shape)
    #oneHotGrad[:, clsResult] = 1
    print('cls result = ', clsResult)
    target = torch.ones(cls.shape[0], dtype=torch.long) * clsResult
    loss = torch.nn.functional.cross_entropy(cls, target)
    loss.backward()
    imgIGrad = batch_input.grad.mean(dim=0)
    imgIGradView = buildGradImage(imgIGrad, nimt - baseline, thresh=thresh)
    imgBasicGrad = batch_input.grad[len(alpha_range) - 1]
    imgBasicGradView=  buildGradImage(imgBasicGrad, nimt, thresh = thresh) 
    croppedImg = toPIL(nimt)
    return imgIGradView, imgBasicGradView, croppedImg, int(clsResult), float(clsScore)


    
if __name__ == "__main__":
    # gradientMethod('dataset/n01443537/11.jpg')

    # img = integratedGradient('dataset/n01443537/11.jpg', sampleSize=100)
    # img[0].show()
    # img[0].save("igrad.png")
    from IPython import embed; embed()
