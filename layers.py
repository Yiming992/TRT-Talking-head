import tensorrt as trt 
import numpy as np 

def get_plugin_creator(plugin_name):
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def Conv7(network, input_tensors, out_channels, kernel_w):
    conv7 = network.add_convolution(input_tensors,out_channels,(7,7),kernel_w)
    conv7.padding = (3,3)
    conv7.stride = (1,1)

    return conv7.get_output(0)

def Conv3(network, input_tensors, out_channels, kernel_w):
    conv3 = network.add_convolution(input_tensors,out_channels,(3,3),kernel_w)
    conv3.padding = (1,1)
    conv3.stride = (1,1) 

    return conv3.get_output(0)

def Conv7Block(network, input_tensors, out_channels, kernel_w, scale, bias):
    conv = Conv7(network,input_tensors,out_channels,kernel_w)
    normalized = InstanceNorm2d(network,conv,scale,bias)
    activation = network.add_activation(input=normalized.get_output(0),type=trt.ActivationType.RELU)
    
    return activation.get_output(0)

def DownsampleBlock(network, input_tensors, in_channels, kernel_w, scale, bias):
    conv = network.add_convolution(input_tensors,in_channels*2,(4,4),kernel_w)
    conv.stride = (2,2)
    conv.padding = (1,1)
    normalized = InstanceNorm2d(network,conv.get_output(0),scale,bias)
    activation = network.add_activation(input=normalized.get_output(0),type=trt.ActivationType.RELU)
    
    return activation.get_output(0)

def UpsampleBlock(network, input_tensors, out_channels, kernel_w, scale, bias):
    convtranspose = network.add_deconvolution(input_tensors,out_channels,(4,4),kernel_w)
    convtranspose.stride = (2,2)
    convtranspose.padding = (1,1)
    normalized = InstanceNorm2d(network,convtranspose.get_output(0),scale,bias)
    activation = network.add_activation(input=normalized.get_output(0),type=trt.ActivationType.RELU)
    
    return activation.get_output(0)

def ResNetBlock(network, input_tensors, num_channels, kernel_ws):
    conv1 = Conv3(network,input_tensors,num_channels,kernel_ws[0])
    normalized1 = InstanceNorm2d(network,conv1,kernel_ws[1],kernel_ws[2])
    activation = network.add_activation(input=normalized1.get_output(0),type=trt.ActivationType.RELU).get_output(0)
    conv2 = Conv3(network,activation,num_channels,kernel_ws[3])
    normalized2 = InstanceNorm2d(network,conv2,kernel_ws[4],kernel_ws[5])
    layer_output = network.add_elementwise(input_tensors,normalized2.get_output(0),trt.ElementWiseOperation.SUM).get_output(0)

    return layer_output

def InstanceNorm2d(network, input_tensors, scale, bias):
    assert scale.shape == bias.shape
    creator = get_plugin_creator("InstanceNormalization_TRT")
    epsilon_field = trt.PluginField("epsilon",np.array([1e-5],dtype=np.float32), trt.PluginFieldType.FLOAT32)
    scale_field = trt.PluginField("scales", scale, trt.PluginFieldType.FLOAT32)
    bias_field = trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32)

    field_collection = trt.PluginFieldCollection([epsilon_field,scale_field,bias_field])
    layer_plugin = creator.create_plugin(name="InstanceNormalization_TRT", field_collection=field_collection)
    
    InstanceNorm2d_layer = network.add_plugin_v2([input_tensors], plugin=layer_plugin)
    
    return InstanceNorm2d_layer

def GridSample(network, input_image, grid, interpolation_mode, padding_mode, align_corners):
    creator = get_plugin_creator("GridSampler")
    
    interpolation_mode_field = trt.PluginField("interpolation_mode",np.array(interpolation_mode,dtype=np.int32),trt.PluginFieldType.INT32)
    padding_mode_field = trt.PluginField("padding_mode",np.array(padding_mode,dtype=np.int32),trt.PluginFieldType.INT32)
    align_corners_field = trt.PluginField("align_corners",np.array(align_corners,dtype=np.int32),trt.PluginFieldType.INT32)
    
    field_collection = trt.PluginFieldCollection([interpolation_mode_field,padding_mode_field,align_corners_field])
    layer_plugin = creator.create_plugin(name="GridSampler",field_collection=field_collection)

    gridsampler = network.add_plugin_v2([input_image,grid], plugin=layer_plugin)

    return gridsampler