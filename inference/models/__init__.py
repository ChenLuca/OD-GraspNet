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
    
    #OSA-Dense Grasp ConvNet

    elif network_name == 'od_1':
        from .OD_ConvNet_1 import Generative_OD_1
        return Generative_OD_1

    #OSA-Dense Grasp ConvNet
    elif network_name == 'od_convnet_1_dilated':
        from .OD_ConvNet_1_dilated import GenerativeOD_dilated
        return GenerativeOD_dilated

    elif network_name == 'od_1_osa_depth_3':
        from .OD_ConvNet_1_OSA_Depth_3 import Generative_OD_1_OSA_Depth_3
        return Generative_OD_1_OSA_Depth_3

    elif network_name == 'od_2':
        from .OD_ConvNet_2 import Generative_OD_2
        return Generative_OD_2


    elif network_name == 'od_3':
        from .OD_ConvNet_3 import Generative_OD_3
        return Generative_OD_3

    elif network_name == 'od_3_osa_depth_10':
        from .OD_ConvNet_3_OSA_Depth_10 import Generative_OD_3_OSA_Depth_10
        return Generative_OD_3_OSA_Depth_10


    elif network_name == 'od_4':
        from .OD_ConvNet_4 import Generative_OD_4
        return Generative_OD_4

    #=====================================================
    
    #OSA-Dense Identity-Mapping Grasp ConvNet

    elif network_name == 'od_1_im':
        from .OD_ConvNet_1_IM import Generative_OD_1_IM
        return Generative_OD_1_IM

    elif network_name == 'od_2_im':
        from .OD_ConvNet_2_IM import Generative_OD_2_IM
        return Generative_OD_2_IM

    elif network_name == 'od_3_im':
        from .OD_ConvNet_3_IM import Generative_OD_3_IM
        return Generative_OD_3_IM

    elif network_name == 'od_4_im':
        from .OD_ConvNet_4_IM import Generative_OD_4_IM
        return Generative_OD_4_IM

    #=====================================================
    
    #OSA-Dense CSP Grasp ConvNet

    elif network_name == 'od_1_csp':
        from .OD_ConvNet_1_CSP import Generative_OD_1_CSP
        return Generative_OD_1_CSP


    elif network_name == 'od_1_csp_fusion_first':
        from .OD_ConvNet_1_CSP_Fusion_First import Generative_OD_1_CSP_Fusion_First
        return Generative_OD_1_CSP_Fusion_First

    elif network_name == 'od_2_csp':
        from .OD_ConvNet_2_CSP import Generative_OD_2_CSP
        return Generative_OD_2_CSP

    elif network_name == 'od_3_csp':
        from .OD_ConvNet_3_CSP import Generative_OD_3_CSP
        return Generative_OD_3_CSP

    elif network_name == 'od_3_csp_osa_depth_10':
        from .OD_ConvNet_3_CSP_OSA_Depth_10 import Generative_OD_3_CSP_OSA_Depth_10
        return Generative_OD_3_CSP_OSA_Depth_10

    elif network_name == 'od_4_csp':
        from .OD_ConvNet_4_CSP import Generative_OD_4_CSP
        return Generative_OD_4_CSP

    #=====================================================
    
    #OSA-Dense IM CSP Grasp ConvNet

    elif network_name == 'od_1_im_csp':
        from .OD_ConvNet_1_IM_CSP import Generative_OD_1_IM_CSP
        return Generative_OD_1_IM_CSP

    elif network_name == 'od_2_im_csp':
        from .OD_ConvNet_2_IM_CSP import Generative_OD_2_IM_CSP
        return Generative_OD_2_IM_CSP

    elif network_name == 'od_3_im_csp':
        from .OD_ConvNet_3_IM_CSP import Generative_OD_3_IM_CSP
        return Generative_OD_3_IM_CSP

    elif network_name == 'od_4_im_csp':
        from .OD_ConvNet_4_IM_CSP import Generative_OD_4_IM_CSP
        return Generative_OD_4_IM_CSP

    #=====================================================
    #OSA-Dense CBAM Grasp ConvNet

    elif network_name == 'odc_1':
        from .ODC_ConvNet_1 import Generative_ODC_1
        return Generative_ODC_1

    elif network_name == 'odc_1_bypass_v2':
        from .ODC_ConvNet_1_Bypass_V2 import Generative_ODC_1_Bypass_V2
        return Generative_ODC_1_Bypass_V2

    elif network_name == 'odc_1_osa_depth_3':
        from .ODC_ConvNet_1_OSA_Depth_3 import Generative_ODC_1_OSA_Depth_3
        return Generative_ODC_1_OSA_Depth_3

    elif network_name == 'odc_1_bypass_v2_osa_depth_3':
        from .ODC_ConvNet_1_Bypass_V2_OSA_Depth_3 import Generative_ODC_1_Bypass_V2_OSA_Depth_3
        return Generative_ODC_1_Bypass_V2_OSA_Depth_3
    
    elif network_name == 'odc_1_bypass_v2_osa_depth_2':
        from .ODC_ConvNet_1_Bypass_V2_OSA_Depth_2 import Generative_ODC_1_Bypass_V2_OSA_Depth_2
        return Generative_ODC_1_Bypass_V2_OSA_Depth_2

    elif network_name == 'odc2_1_bypass_v2':
        from .ODC2_ConvNet_1_Bypass_V2 import Generative_ODC2_1_Bypass_V2
        return Generative_ODC2_1_Bypass_V2

    #=====================================================


    #OSA-Dense RFB Grasp ConvNet

    elif network_name == 'odr_1':
        from .ODR_ConvNet_1 import Generative_ODR_1
        return Generative_ODR_1

    elif network_name == 'odr_2':
        from .ODR_ConvNet_2 import Generative_ODR_2
        return Generative_ODR_2

    elif network_name == 'odr_3':
        from .ODR_ConvNet_3 import Generative_ODR_3
        return Generative_ODR_3

    elif network_name == 'odr_4':
        from .ODR_ConvNet_4 import Generative_ODR_4
        return Generative_ODR_4
    
    #=====================================================

    #OSA-Dense RFB IM Grasp ConvNet

    elif network_name == 'odr_1_im':
        from .ODR_ConvNet_1_IM import Generative_ODR_1_IM
        return Generative_ODR_1_IM

    elif network_name == 'odr_2_im':
        from .ODR_ConvNet_2_IM import Generative_ODR_2_IM
        return Generative_ODR_2_IM

    elif network_name == 'odr_3_im':
        from .ODR_ConvNet_3_IM import Generative_ODR_3_IM
        return Generative_ODR_3_IM

    elif network_name == 'odr_4_im':
        from .ODR_ConvNet_4_IM import Generative_ODR_4_IM
        return Generative_ODR_4_IM
    
    #=====================================================

    #OSA-Dense RFB CSP Grasp ConvNet

    elif network_name == 'odr_1_csp':
        from .ODR_ConvNet_1_CSP import Generative_ODR_1_CSP
        return Generative_ODR_1_CSP

    elif network_name == 'odr_1_csp_fusion_first':
        from .ODR_ConvNet_1_CSP_Fusion_First import Generative_ODR_1_CSP_Fusion_First
        return Generative_ODR_1_CSP_Fusion_First

    elif network_name == 'odr_2_csp':
        from .ODR_ConvNet_2_CSP import Generative_ODR_2_CSP
        return Generative_ODR_2_CSP

    elif network_name == 'odr_3_csp':
        from .ODR_ConvNet_3_CSP import Generative_ODR_3_CSP
        return Generative_ODR_3_CSP

    elif network_name == 'odr_3_csp_osa_depth_10':
        from .ODR_ConvNet_3_CSP_OSA_Depth_10 import Generative_ODR_3_CSP_OSA_Depth_10
        return Generative_ODR_3_CSP_OSA_Depth_10

    elif network_name == 'odr_4_csp':
        from .ODR_ConvNet_4_CSP import Generative_ODR_4_CSP
        return Generative_ODR_4_CSP
    
    #=====================================================
    
    #OSA-Dense RFB IM CSP Grasp ConvNet

    elif network_name == 'odr_1_im_csp':
        from .ODR_ConvNet_1_IM_CSP import Generative_ODR_1_IM_CSP
        return Generative_ODR_1_IM_CSP

    elif network_name == 'odr_1_im_csp_max':
        from .ODR_ConvNet_1_IM_CSP_MAX import Generative_ODR_1_IM_CSP_MAX
        return Generative_ODR_1_IM_CSP_MAX

    elif network_name == 'odr_2_im_csp':
        from .ODR_ConvNet_2_IM_CSP import Generative_ODR_2_IM_CSP
        return Generative_ODR_2_IM_CSP

    elif network_name == 'odr_3_im_csp':
        from .ODR_ConvNet_3_IM_CSP import Generative_ODR_3_IM_CSP
        return Generative_ODR_3_IM_CSP

    elif network_name == 'odr_4_im_csp':
        from .ODR_ConvNet_4_IM_CSP import Generative_ODR_4_IM_CSP
        return Generative_ODR_4_IM_CSP

    #=====================================================

    #OSA-Dense CBAM CSP Grasp ConvNet
    elif network_name == 'odc_1_csp':
        from .ODC_ConvNet_1_CSP import Generative_ODC_1_CSP
        return Generative_ODC_1_CSP

    elif network_name == 'odc_1_csp_osa_depth_5_bypass_v2':
        from .ODC_ConvNet_1_CSP_OSA_Depth_5_Bypass_V2 import Generative_ODC_1_CSP_OSA_Depth_5_Bypass_V2
        return Generative_ODC_1_CSP_OSA_Depth_5_Bypass_V2

    elif network_name == 'odc_3_csp_osa_depth_10':
        from .ODC_ConvNet_3_CSP_OSA_Depth_10 import Generative_ODC_3_CSP_OSA_Depth_10
        return Generative_ODC_3_CSP_OSA_Depth_10

    elif network_name == 'odc_3_csp_osa_depth_10_bypass':
        from .ODC_ConvNet_3_CSP_OSA_Depth_10_Bypass import Generative_ODC_3_CSP_OSA_Depth_10_Bypass
        return Generative_ODC_3_CSP_OSA_Depth_10_Bypass

    elif network_name == 'odc_3_csp_osa_depth_5_bypass_v2':
        from .ODC_ConvNet_3_CSP_OSA_Depth_5_Bypass_V2 import Generative_ODC_3_CSP_OSA_Depth_5_Bypass_V2
        return Generative_ODC_3_CSP_OSA_Depth_5_Bypass_V2

    elif network_name == 'odc_3_csp_osa_depth_10_bypass_v2':
        from .ODC_ConvNet_3_CSP_OSA_Depth_10_Bypass_V2 import Generative_ODC_3_CSP_OSA_Depth_10_Bypass_V2
        return Generative_ODC_3_CSP_OSA_Depth_10_Bypass_V2
    #=====================================================
    
    # Not good !!
    #OSA-Dense RFB CBAM CSP Grasp ConvNet

    elif network_name == 'odrc_3_csp_osa_depth_10_bypass_v2':
        from .ODRC_ConvNet_3_CSP_OSA_Depth_10_Bypass_V2 import Generative_ODRC_3_CSP_OSA_Depth_10_Bypass_V2
        return Generative_ODRC_3_CSP_OSA_Depth_10_Bypass_V2
    #=====================================================
        # not that strange !!
    elif network_name == 'real_strange':
        from .osadense_grconvnet import GenerativeOSA_Strange
        return GenerativeOSA_Strange
    
    # not that strange !!
    elif network_name == 'strange':
        from .osadense_graspnet import GenerativeOSADense
        return GenerativeOSADense
    #=====================================================

    elif network_name == "od_4_cbam":
        from .OD_ConvNet_4_CBAM import Generative_OD_4_CBAM
        return Generative_OD_4_CBAM

    elif network_name == "od_4_cbam_2":
        from .OD_ConvNet_4_CBAM_2 import Generative_OD_4_CBAM_2
        return Generative_OD_4_CBAM_2

    elif network_name == "odc_shuffle_4":
        from .ODC_Shuffle_ConvNet_4 import Generative_ODC_Shuffle_4
        return Generative_ODC_Shuffle_4
    
    elif network_name == "odc_shuffle_v2_4":
        from .ODC_Shuffle_v2_ConvNet_4 import Generative_ODC_Shuffle_v2_4
        return Generative_ODC_Shuffle_v2_4

    elif network_name == "odc_shuffle_v2_1":
        from .ODC_Shuffle_v2_ConvNet_1 import Generative_ODC_Shuffle_v2_1
        return Generative_ODC_Shuffle_v2_1

    elif network_name == "odc_shuffle_v2_4_bypass":
        from .ODC_Shuffle_v2_ConvNet_4_bypass import Generative_ODC_Shuffle_v2_4_bypass
        return Generative_ODC_Shuffle_v2_4_bypass

    elif network_name == "odc_shuffle_v3_4":
        from .ODC_Shuffle_v3_ConvNet_4 import Generative_ODC_Shuffle_v3_4
        return Generative_ODC_Shuffle_v3_4

    elif network_name == "odc_shuffle_v4_4":
        from .ODC_Shuffle_v4_ConvNet_4 import Generative_ODC_Shuffle_v4_4
        return Generative_ODC_Shuffle_v4_4

    elif network_name == "odc_shuffle_unet_4":
        from .ODC_Shuffle_UNet_ConvNet_4 import Generative_ODC_Shuffle_UNet_4
        return Generative_ODC_Shuffle_UNet_4

    elif network_name == "odc_shuffle_unet_1":
        from .ODC_Shuffle_UNet_ConvNet_1 import Generative_ODC_Shuffle_UNet_1
        return Generative_ODC_Shuffle_UNet_1

    elif network_name == "odc_im_shuffle_unet_1":
        from .ODC_IM_Shuffle_UNet_ConvNet_1 import Generative_ODC_IM_Shuffle_UNet_1
        return Generative_ODC_IM_Shuffle_UNet_1

    elif network_name == "odc_shuffle_unet_4_cbamfirst":
        from .ODC_Shuffle_UNet_ConvNet_4_CbamFirst import Generative_ODC_Shuffle_UNet_4_CbamFirst
        return Generative_ODC_Shuffle_UNet_4_CbamFirst

    elif network_name == "odc_shuffle_1":
        from .ODC_Shuffle_ConvNet_1 import Generative_ODC_Shuffle_1
        return Generative_ODC_Shuffle_1

    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
