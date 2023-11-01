import os
import yaml
import logging


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


class CONFIG(object):
    '''
    Stores all configures
    '''
    def __init__(self, input=None, mode='train'):
        '''
        Loads config file
        :param input (str): path to config file
        :return:
        '''
        self.input_file_path = input
        self.config = self.read_to_dict(input)
        self._logger, self._save_path = self.load_logger(mode)

        # update save_path to config file
        self.update_config(log={'path': self._save_path})

        # initiate device environments
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['device']['gpu_ids']

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def load_logger(self, mode='train'):
        # set save path
        if mode == 'train':
            save_path = os.path.join(self.config['save_root_path'], self.config['exp_name'])
        else:
            save_path = os.path.join(self.config['save_root_path'], self.config['exp_name'], 'out')
        os.makedirs(save_path, exist_ok=True)

        # set logging
        logfile = os.path.join(save_path, 'log.txt')
        file_handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.__file_handler = file_handler

        # configure logger
        logger = logging.getLogger('Empty')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        return logger, save_path

    def log_string(self, content):
        self._logger.info(content)
        print(content)

    def read_to_dict(self, input):
        if not input:
            return dict()
        if isinstance(input, str) and os.path.isfile(input):
            if input.endswith('yaml'):
                with open(input, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                ValueError('Config file should be with the format of *.yaml')
        elif isinstance(input, dict):
            config = input
        else:
            raise ValueError('Unrecognized input type (i.e. not *.yaml file nor dict).')

        return config

    def update_config(self, *args, **kwargs):
        '''
        update config and corresponding logger setting
        :param input: dict settings add to config file
        :return:
        '''
        cfg1 = dict()
        for item in args:
            cfg1.update(self.read_to_dict(item))

        cfg2 = self.read_to_dict(kwargs)

        new_cfg = {**cfg1, **cfg2}

        update_recursive(self.config, new_cfg)
        # when update config file, the corresponding logger should also be updated.
        self.__update_logger()

    def write_config(self):
        # save training config
        output_file = os.path.join(self._save_path, 'out_config.yaml')

        import shutil
        shutil.copyfile(self.input_file_path, output_file)

    def __update_logger(self):
        # configure logger
        name = self.config['mode'] if 'mode' in self.config else self._logger.name
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.__file_handler)
        self._logger = logger
