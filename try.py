import nbox

# As all these models come from the popular frameworks you use on daily basis
# such as torchvision or efficient_pytorch they have same parameters you can pass to the load function 
model = nbox.load("mobilenetv2")
# If you want to use the pre trained version
# model = nbox.load("mobilenetv2". pretrained=True)
out = model('./tests/assets/cat.jpg')
print(out)
