import aibox

model = aibox.load("mobilenetv2")
out = model("./nimblebox.png")
print(out)