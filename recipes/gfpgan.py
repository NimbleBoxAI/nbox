import os
import numpy as np
from PIL import Image

if not torch.cuda.is_available():  # CPU
  import warnings
  warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                'If you really want to use it, please modify the corresponding codes.')
  bg_upsampler = None
else:
  from basicsr.archs.rrdbnet_arch import RRDBNet
  from realesrgan import RealESRGANer
  model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
  bg_upsampler = RealESRGANer(
    scale=2,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    model=model,
    tile=args.bg_tile,
    tile_pad=10,
    pre_pad=0,
    half=True
  )  # need to set False in CPU mode

class GFPGAN():
  def __init__(self, model_path, upscale, arch, channel):
    self.restorer = GFPGANer(
      model_path= model_path,
      upscale= upscale,
      arch= arch,
      channel_multiplier= channel,
      bg_upsampler=bg_upsampler
    )

  def __call__(self, img_path, aligned, only_center_face, paste_back):
    # read image
    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = Image.open(img_path)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = self.restorer.enhance(
      input_img,
      has_aligned = aligned,
      only_center_face = only_center_face,
      paste_back = paste_back
    )

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
      # save cropped face
      save_crop_path = os.path.join(args.save_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
      imwrite(cropped_face, save_crop_path)
      # save restored face
      if args.suffix is not None:
        save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
      else:
        save_face_name = f'{basename}_{idx:02d}.png'
      save_restore_path = os.path.join(args.save_root, 'restored_faces', save_face_name)
      imwrite(restored_face, save_restore_path)
      # save comparison image
      cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
      imwrite(cmp_img, os.path.join(args.save_root, 'cmp', f'{basename}_{idx:02d}.png'))


def main(
  upscale = 2, # The final upsampling scale of the image
  arch = 'clean', # The GFPGAN architecture. Option: clean | original
  channel = 2, # Channel multiplier for large networks of StyleGAN2
  model_path = 'experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth', # The path to the GFPGAN model
  bg_upsampler = 'realesrgan', # background upsampler
  bg_tile = 400, # Tile size for background sampler, 0 for no tile during testing
  test_path = 'inputs/whole_imgs', # Input folder
  suffix = None, # Suffix of the restored faces
  only_center_face = False, # Only restore the center face
  aligned = False, # Input are aligned faces
  paste_back = True, # Paste the restored faces back to images
  save_root = 'results', # Path to save root
  ext = 'auto', # Image extension. Options: auto | jpg | png, auto means using the same extension as inputs
):
  pass