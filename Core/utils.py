import matplotlib.pyplot as plt
import os

def rescale_for_plotting(tensor):
    tensor = tensor.detach().cpu().numpy().transpose(1,2,0)
    tensor = (tensor - tensor.min())/(tensor.max() - tensor.min())
    return tensor
def plot_sample(hr,lr,sr):
    '''
    Plots full resolution and low resolution samples side by side.
    Each sample input contains a (full_resolution, low_resolution) iterable
    Args:
        sample: sample from dataset
    '''
    full_resolution = rescale_for_plotting(hr)
    super_resolution = rescale_for_plotting(sr)
    low_resolution = rescale_for_plotting(lr)

    f,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(low_resolution)
    ax1.set_title('low resolution')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(super_resolution)
    ax2.set_title('super resolution')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(full_resolution)
    ax3.set_title('full resolution')
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.close(f)
    return f

def create_folder(PATH):
    if os.path.isdir(PATH):
        return None
    else:
        os.system("mkdir -p {}".format(PATH))