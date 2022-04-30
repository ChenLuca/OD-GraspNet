def get_network(network_name):
    network_name = network_name.lower()
    
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet

    #=====================================================
    
    # strange !!
    elif network_name == 'od_fi':
        from .OD_FI import GenerativeOD_Fusion_Internal
        return GenerativeOD_Fusion_Internal

    elif network_name == 'odc_fi':
        from .ODC_FI import GenerativeODC_Fusion_Internal
        return GenerativeODC_Fusion_Internal
    
    # not that strange !!
    elif network_name == 'od':
        from .OD import GenerativeOD
        return GenerativeOD

    elif network_name == 'odc':
        from .ODC import GenerativeODC
        return GenerativeODC
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
