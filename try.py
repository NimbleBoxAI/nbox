import aibox

model = aibox.load("mobilenetv2")
out = model('./tests/assets/cat.jpg')