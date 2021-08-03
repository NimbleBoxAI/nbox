import nbox

model = nbox.load("mobilenetv2")
out = model('./tests/assets/cat.jpg')
print(out)