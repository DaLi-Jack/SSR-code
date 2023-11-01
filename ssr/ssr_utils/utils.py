import torch
import numpy as np
import os
import urllib
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# replace for torch.repeat_interleave
def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

def read_obj_point(obj_path):
    with open(obj_path, 'r') as f:
        content_list = f.readlines()
        point_list = [line.rstrip("\n").lstrip("v ").split(" ") for line in content_list]
        for point in point_list:
            for i in range(3):
                point[i] = float(point[i])
        return np.array(point_list)

def write_obj_point(points,obj_path):
    with open(obj_path,'w') as f:
        for i in range(points.shape[0]):
            point=points[i]
            write_line="v %.4f %.4f %.4f\n"%(point[0],point[1],point[2])
            f.write(write_line)
    return

def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')

    if 'net' in ckpt:
        ckpt = ckpt['net']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    if 'encoder.latent' in modified:
        del modified['encoder.latent']

    model.load_state_dict(modified)
    return model

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, **kwargs):
        '''
        initialize model and optimizer.
        :param cfg: configuration file
        :param kwargs: model, optimizer and other specs.
        '''
        self.cfg = cfg
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'iter':0, 'min_loss': 1e8})
        self._saved_filename = 'model_last.pth'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        outdict = kwargs
        for k, v in self._module_dict.items():
            if hasattr(v, 'state_dict'):
                outdict[k] = v.state_dict()
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.cfg.config['log']['path'], filename))

    def load(self, filename, *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''

        if self.is_url(filename):
            return self.load_url(filename, *domain)
        else:
            return self.load_file(filename, *domain)

    def parse_checkpoint(self):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config['resume']:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.log_string('Begin to resume from the last checkpoint.')
            self.resume()
        elif self.cfg.config['finetune']:
            # only load net weights.
            self.cfg.log_string('Begin to finetune from the existing weight.')
            self.finetune()
        else:
            self.cfg.log_string('Begin to train from scratch.')

    def finetune(self):
        '''
        finetune from existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config['weight'], str):
            weight_paths = [self.cfg.config['weight']]
        else:
            weight_paths = self.cfg.config['weight']

        for weight_path in weight_paths:
            if not os.path.exists(weight_path):
                self.cfg.log_string('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, 'net')
                self.cfg.log_string('Weights for finetuning loaded.')

    def resume(self):
        '''
        resume the lastest checkpoint
        :return:
        '''
        checkpoint_root = os.path.dirname(self.cfg.save_path)
        saved_log_paths = os.listdir(checkpoint_root)
        saved_log_paths.sort(reverse=True)

        for last_path in saved_log_paths:
            last_checkpoint = os.path.join(checkpoint_root, last_path, self.saved_filename)
            if not os.path.exists(last_checkpoint):
                continue
            else:
                self.load(last_checkpoint)
                self.cfg.log_string('Last checkpoint resumed.')
                return

        self.cfg.log_string('Warning: resume failed: No checkpoint available. Begin to train from scratch.')

    def load_file(self, filename, *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''

        if os.path.exists(filename):
            self.cfg.log_string('Loading checkpoint from %s.' % (filename))
            checkpoint = torch.load(filename)
            if "LDIF" in filename:
                new_checkpoint={"net":{}}
                for k,v in checkpoint["net"].items():
                    new_k="mesh_reconstruction."+k
                    new_checkpoint["net"][new_k]=v
                scalars = self.parse_state_dict(new_checkpoint, *domain)
            else:

                scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.log_string('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''
        for key, value in self._module_dict.items():
            if key=="opt":
                #print(self._module_dict[key])
                print("skipping optimizer")
                continue
            if key=="sch":
                print("skipping scheduler")
                continue

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if hasattr(value, 'load_state_dict'):
                    '''load weights module by module'''

                    value.load_state_dict(checkpoint[key])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.log_string('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in our models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.log_string('Warning: the remaining modules %s in checkpoint are not found in our current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars

def get_optimizer(config, net):
    '''
    get optimizer for networks
    :param config: configuration file
    :param model: nn.Module network
    :return:
    '''
    if config['optimizer']['type'] == 'AdamW':
        if config["method"]=="depth_estimation":
            params = [{"params": net.get_1x_lr_params(), "lr": float(config['optimizer']['lr'] / 10)},
                      {"params": net.get_10x_lr_params(), "lr": float(config['optimizer']['lr'])}]

            optimizer = torch.optim.AdamW(params, lr=float(config['optimizer']['lr']),
                                          weight_decay=config['optimizer']['weight_decay'])

        else:
            '''collect parameters with specific optimizer spec'''
            optimizer = torch.optim.AdamW(net.parameters(),lr=float(config['optimizer']['lr']),
                                         weight_decay=config['optimizer']['weight_decay'])
    elif config["optimizer"]["type"] == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),lr=float(config["optimizer"]["lr"]),
                                    weight_decay=config["optimizer"]["weight_decay"],
                                    momentum=config["optimizer"]["momentum"])
    elif config["optimizer"]["type"]=="Adam":
        if config["method"]=="depth_estimation":
            params = [{"params": net.get_1x_lr_params(), "lr": float(config['optimizer']['lr'] / 10)},
                      {"params": net.get_10x_lr_params(), "lr": float(config['optimizer']['lr'])}]
            optimizer = torch.optim.Adam(params, lr=float(config["optimizer"]["lr"]),
                                         betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]))
        else:
            eps=config["optimizer"]["eps"]
            weight_decay=config["optimizer"]["weight_decay"]
            optimizer = torch.optim.Adam(net.parameters(), lr=float(config["optimizer"]["lr"]),
                                         betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),eps=eps,weight_decay=weight_decay)
    elif config["optimizer"]["type"]=="RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(),lr=float(config["optimizer"]["lr"]),
                                        weight_decay=config["optimizer"]["weight_decay"],
                                        momentum=config["optimizer"]["momentum"])
    return optimizer


def load_scheduler(config,optimizer,train_loader):
    if config["scheduler"]["type"]=="OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config["optimizer"]["lr"], epochs=config["other"]["nepoch"], steps_per_epoch=len(train_loader),
                                                  cycle_momentum=True,
                                                  base_momentum=0.85, max_momentum=0.95, last_epoch=config["scheduler"]['last_epoch'],
                                                  div_factor=config["scheduler"]['div_factor'],
                                                  final_div_factor=config['scheduler']['final_div_factor'])
    elif config["scheduler"]["type"]== "MultiStepLR":
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,config["scheduler"]["milestone"],gamma=config['scheduler']['gamma'])
    elif config["scheduler"]["type"]=="ReduceLROnPlateau":
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=float(config['scheduler']['factor']),
                                                               patience=config['scheduler']['patience'],
                                                               threshold=float(config['scheduler']['threshold']),
                                                               verbose=True)
    elif config["scheduler"]["type"]=="CosineAnnealingLR":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["other"]["nepoch"],eta_min=config["scheduler"]["eta_min"])
    
    return scheduler

def load_device(cfg):
    '''
    load device settings
    :param config:
    :return:
    '''
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        return torch.device("cuda")
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")

def get_model(cfg,device):
    if cfg['method']=="ssr":
        from ssr.model.network import SSRNet
        model=SSRNet(cfg)
    else:
        raise NotImplementedError
    if cfg['model']['stop_encoder_grad']:
        print("Encoder frozen, stop_encoder_grad")
        model.encoder.eval()
    return model

def get_loss(cfg, mode):
    if cfg['method']=="ssr":
        from ssr.model.loss import MonoSDFLoss
        loss=MonoSDFLoss(cfg, mode)
    else:
        raise NotImplementedError
    return loss

def get_dataloader(cfg,mode):
    if cfg['data']['dataset'] == 'FRONT3D':
        from ssr.dataloader.front3d_dataloader import Front3D_Recon_dataloader
        dataloader=Front3D_Recon_dataloader(cfg,mode)
    elif cfg['data']['dataset'] == 'Pix3D':
        from ssr.dataloader.pix3d_dataloader import Pix3D_Recon_dataloader
        dataloader=Pix3D_Recon_dataloader(cfg,mode)
    elif cfg['data']['dataset'] == 'SUNRGBD':
        from ssr.dataloader.sunrgbd_dataloader import SUNRGBD_Recon_dataloader
        dataloader=SUNRGBD_Recon_dataloader(cfg,mode)
    else:
        raise NotImplementedError
    return dataloader

def get_trainer(config):
    if config["method"]=="ssr":
        from ssr.trainer import Recon_trainer
        trainer=Recon_trainer
    else:
        raise NotImplementedError
    return trainer
