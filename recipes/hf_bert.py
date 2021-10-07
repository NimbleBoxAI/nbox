import nbox

# the model key for transformers has structure
# transformers/<model_key>::<model_arch>
model_key = "transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM"
model = nbox.load(model_key)

# use the model
out = model("This is a string given to the model. You do not need to concern "
  "yourself with the tokenization procedure. All of the gets handled internally")

# deploy the model
url, key = model.deploy(
  "hello world",
  wait_for_deployment=True
)
