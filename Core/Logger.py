from torch.utils.tensorboard import SummaryWriter

class Logger:
    '''
    Class to handle tensroboard summary writer
    '''
    def __init__(self,config):
        self.writer = SummaryWriter(config.exp_name)