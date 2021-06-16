


class Config(object):
    input_size = (178, 218, 3)
    lr = 1e-4
    batch_size = 64

    def __init__(self,):
        self.expand_layer_filter_rate = 4
        self.dw_seperable_layer_filter_rate = 3
        self.conv_filter_size = 96
        self.conv_filter = (7, 7)
        self.conv_stride = (2, 2)
        self.max_pool_filter = (3, 3)
        self.max_pool_stride = (2, 2)
        self.slim_module_filter_sizes = [16, 32, 48, 64]
        
        self.layers = {}
        for i in range(4):
            
            self.sse_filters = {"squeeze": self.slim_module_filter_sizes[i], 
                                "expand_1": self.slim_module_filter_sizes[i] * self.expand_layer_filter_rate, 
                                "expand_3": self.slim_module_filter_sizes[i] * self.expand_layer_filter_rate,
                                "dw_seperable": self.slim_module_filter_sizes[i] * self.dw_seperable_layer_filter_rate}
            self.layers["layer" + str(i+1)] = self.sse_filters 