from torchinfo.layer_info import LayerInfo

class LayerHelper:
    @staticmethod
    def extract_params(layer: LayerInfo):
        q = layer.output_size[2]
        p = layer.output_size[3]
        c = layer.input_size[1]            # input channels
        n = layer.input_size[0]            # input batch_size
        m = layer.output_size[1]
        s = layer.module.kernel_size[0]
        r = layer.module.kernel_size[1]
        wpad = layer.module.padding[0]
        hpad = layer.module.padding[1]
        wstride = layer.module.stride[0]
        hstride = layer.module.stride[1]

        return q, p, c, n, m, s, r, wpad, hpad, wstride, hstride

