import torch
import pyredner
import h5py

from matplotlib.pyplot import imsave



# Load the Basel face model
with h5py.File(r'/content/drive/MyDrive/Colab_Notebooks/3laba/model2017-1_bfm_nomouth.h5', 'r') as hf:
    shape_mean = torch.tensor(hf['shape/model/mean'], 
                              device = pyredner.get_device())
    shape_basis = torch.tensor(hf['shape/model/pcaBasis'], 
                               device = pyredner.get_device())
    triangle_list = torch.tensor(hf['shape/representer/cells'], 
                                 device = pyredner.get_device())
    color_mean = torch.tensor(hf['color/model/mean'], 
                              device = pyredner.get_device())
    color_basis = torch.tensor(hf['color/model/pcaBasis'], 
                               device = pyredner.get_device())


indices = triangle_list.permute(1, 0).contiguous()

def model(
        cam_pos, 
        cam_look_at, 
        shape_coeffs, 
        color_coeffs, 
        ambient_color, 
        dir_light_intensity):
    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)
    normals = pyredner.compute_vertex_normal(vertices, indices)
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)
    m = pyredner.Material(use_vertex_color = True)
    obj = pyredner.Object(vertices = vertices, 
                          indices = indices, 
                          normals = normals, 
                          material = m, 
                          colors = colors)
    cam = pyredner.Camera(position = cam_pos,
                          # Center of the vertices                          
                          look_at = cam_look_at,
                          up = torch.tensor([0.0, 1.0, 0.0]),
                          fov = torch.tensor([45.0]),
                          resolution = (256, 256))
    scene = pyredner.Scene(camera = cam, objects = [obj])
    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(torch.tensor([0.0, 0.0, -1.0]), 
                                          dir_light_intensity)
    img = pyredner.render_deferred(scene = scene, 
                                   lights = [ambient_light, dir_light])
    return img

cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277])
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918])
img = model(cam_pos, 
            cam_look_at, 
            torch.zeros(199, device = pyredner.get_device()),
            torch.zeros(199, device = pyredner.get_device()),
            torch.ones(3), 
            torch.zeros(3))

imsave('/content/drive/MyDrive/Colab_Notebooks/3laba/defolt.png',torch.pow(img, 1.0/2.2).cpu().numpy())
# imshow(torch.pow(img, 1.0/2.2).cpu())

target = pyredner.imread('/content/drive/MyDrive/Colab_Notebooks/3laba/target.png').to(pyredner.get_device())

# imshow(torch.pow(target, 1.0/2.2).cpu())


# Set requires_grad=True since we want to optimize them later
cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277], 
                       requires_grad=True)
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918], 
                           requires_grad=True)
shape_coeffs = torch.zeros(199, device = pyredner.get_device(), 
                           requires_grad=True)
color_coeffs = torch.zeros(199, device = pyredner.get_device(), 
                           requires_grad=True)
ambient_color = torch.ones(3, device = pyredner.get_device(), 
                           requires_grad=True)
dir_light_intensity = torch.zeros(3, device = pyredner.get_device(), 
                                  requires_grad=True)

# Use two different optimizers for different learning rates
optimizer = torch.optim.Adam(
                             [
                              shape_coeffs, 
                              color_coeffs, 
                              ambient_color, 
                              dir_light_intensity], 
                             lr=0.1)
cam_optimizer = torch.optim.Adam([cam_pos, cam_look_at], lr=0.5)


# Run 500 Adam iterations
num_iters = 500
for t in range(num_iters):
    optimizer.zero_grad()
    cam_optimizer.zero_grad()
    img = model(cam_pos, cam_look_at, shape_coeffs, 
                color_coeffs, ambient_color, dir_light_intensity)
    # Compute the loss function. Here it is L2 plus a regularization 
    # term to avoid coefficients to be too far from zero.
    # Both img and target are in linear color space, 
    # so no gamma correction is needed.

    loss = (img - target).pow(2).mean()
    loss = loss \
         + 0.0001 * shape_coeffs.pow(2).mean() \
         + 0.001 * color_coeffs.pow(2).mean()
    loss.backward()

    optimizer.step()
    cam_optimizer.step()

    ambient_color.data.clamp_(0.0)
    dir_light_intensity.data.clamp_(0.0)


    # Only store images every 10th iterations
    if t % 10 == 0:
        print(t, num_iters, loss.data.item())
        imsave(f'/content/drive/MyDrive/Colab_Notebooks/3laba/train_{t}.png',torch.pow(img.data, 1.0/2.2).cpu().numpy())


























