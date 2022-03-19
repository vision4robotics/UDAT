from .uavdark70 import UAVDARK70Dataset
from .nat import NATDataset
from .uav import UAVDataset
from .nat_l import NAT_LDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAVDark70' == name:
            dataset = UAVDARK70Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'NAT' == name:
            dataset = NATDataset(**kwargs)
        elif 'NAT_L' == name:
            dataset = NAT_LDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

