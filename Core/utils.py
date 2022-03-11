import matplotlib.pyplot as plt
import os

def plot_for_dataset_sample(sample):
    '''
    Plots full resolution and low resolution samples side by side.
    Each sample input contains a (full_resolution, low_resolution) iterable
    Args:
        sample: sample from dataset
    '''
    full_resolution = sample[0].cpu().numpy().transpose(1,2,0)
    low_resolution = sample[1].cpu().numpy().transpose(1,2,0)
    f,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(full_resolution)
    ax1.set_title('Full resolution')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.imshow(low_resolution)
    ax2.set_title('Low resolution')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()

def create_folder(PATH):
    if os.path.isdir(PATH):
        return None
    else:
        os.system("mkdir -p {}".format(PATH))