from .uav import UAVDataset
from .UAVDark70 import UAVDark70Dataset
from .nat import NATDataset
from .nat_l import NAT_LDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'UAVDark70', 'UAV', 'darktrack', 'darktrack'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAVDark70' == name:
            dataset = UAVDark70Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'NAT' == name:
            dataset = NATDataset(**kwargs)
        elif 'NAT_L' == name:
            dataset = NAT_LDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

