import argparse
import tensorrt as trt 
from layers import Conv7Block, DownsampleBlock, UpsampleBlock, Conv3, Conv7, ResNetBlock, GridSample
import numpy as np 
import ctypes
import torch
import logging
import yaml

'''
input and output flow of the network

source_image, morph_params --> morphed image
morphed_image[0], base_grid, rotate_params --> rotated_images
rotated_images[0], rotated_images[1], combine_params -->combined_image
'''
class EncoderDecoder:
    # Implement a encoder-decoder module with residula bottleneck blocks
    def __init__(self,
                 output_channels,
                 bottleneck_block_count,                                                                                                                         
                 kernel_ws):        
        self.output_channels = output_channels 
        self.bottleneck_block_count = bottleneck_block_count
        self.kernel_ws = kernel_ws 
    
    def __call__(self, network, input_tensor, weight_index):
        current_channels = self.output_channels
    
        input_tensor = Conv7Block(network,input_tensor,self.output_channels,self.kernel_ws[weight_index],
                                  self.kernel_ws[weight_index+1],self.kernel_ws[weight_index+2])
        weight_index += 3

        for i in range(3):
            input_tensor = DownsampleBlock(network,input_tensor,current_channels,self.kernel_ws[weight_index],
                                           self.kernel_ws[weight_index+1],self.kernel_ws[weight_index+2])
            current_channels *= 2
            weight_index += 3
            
        for i in range(self.bottleneck_block_count):
            input_tensor = ResNetBlock(network,input_tensor,current_channels,self.kernel_ws[weight_index:(weight_index+6)])
            weight_index += 6  

        for i in range(3):
            input_tensor = UpsampleBlock(network,input_tensor,current_channels//2,self.kernel_ws[weight_index],self.kernel_ws[weight_index+1],
                                        self.kernel_ws[weight_index+2])
            current_channels //= 2
            weight_index += 3

        return weight_index,input_tensor   
     
class UNetModule:
    # Implement a UNet module with residual bottleneck blocks
    def __init__(self,
                 output_channels,
                 bottleneck_block_count,
                 kernel_ws):
        self.output_channels = output_channels 
        self.bottleneck_block_count = bottleneck_block_count 
        self.kernel_ws = kernel_ws 

    def __call__(self, network, input_tensor, weight_index):
        current_channels = self.output_channels
        downward_outputs = []

        input_tensor = Conv7Block(network,input_tensor,self.output_channels,self.kernel_ws[weight_index],
                                  self.kernel_ws[weight_index+1],self.kernel_ws[weight_index+2])
        downward_outputs.insert(0,input_tensor)
        weight_index += 3
        upsample_layer_index = 0
        
        for i in range(3):
            input_tensor = DownsampleBlock(network,input_tensor,current_channels,self.kernel_ws[weight_index],
                                            self.kernel_ws[weight_index+1],self.kernel_ws[weight_index+2])
            current_channels *= 2
            weight_index += 3
            if i < 2:
                downward_outputs.insert(0,input_tensor)
      
        for i in range(self.bottleneck_block_count):
            input_tensor = ResNetBlock(network,input_tensor,current_channels,self.kernel_ws[weight_index:(weight_index+6)])
            weight_index += 6
        
        for i in range(3):
            current_index = weight_index + (3 - upsample_layer_index) * 3
            input_tensor = UpsampleBlock(network,input_tensor,current_channels//2,self.kernel_ws[current_index],
                                            self.kernel_ws[current_index+1],self.kernel_ws[current_index+2])
            input_tensor = network.add_concatenation([input_tensor,downward_outputs[upsample_layer_index]]).get_output(0)
            current_channels = current_channels // 2
            upsample_layer_index += 1

        input_tensor = Conv7Block(network,input_tensor,current_channels,self.kernel_ws[weight_index],
                                  self.kernel_ws[weight_index+1],self.kernel_ws[weight_index+2])

        weight_index += 12
        return weight_index,input_tensor
             

class Morpher:
    # Implement the face morpher network
    def __init__(self,
                 image_channels,
                 pose_size,
                 intermediate_channels,
                 bottleneck_block_count,
                 kernel_ws):
        self.image_channels = image_channels 
        self.pose_size = pose_size
        self.intermediate_channels = intermediate_channels
        self.bottleneck_block_count = bottleneck_block_count
        self.kernel_ws = kernel_ws
        self.weight_index = 0

    def __call__(self, network):
        input_image = network.add_input(name="input_image",dtype=trt.float32,shape=(-1,4,-1,-1))
        input_pose = network.add_input(name="input_pose",dtype=trt.float32,shape=(-1,self.pose_size))

        input_shape = network.add_shape(input_image).get_output(0)
        shape_offset = network.add_constant([4],np.array([0,1,0,0],dtype=np.int32)).get_output(0)
        input_shape = network.add_elementwise(input_shape,shape_offset,trt.ElementWiseOperation.SUB).get_output(0)

        shuffle_layer = network.add_shuffle(input_pose)
        shuffle_layer.reshape_dims = (0,0,1,1)
        
        slice_layer = network.add_slice(shuffle_layer.get_output(0),(0,0,0,0),(0,0,0,0),(1,1,0,0))
        slice_layer.set_input(2,input_shape)
        
        input_tensor = network.add_concatenation([input_image, slice_layer.get_output(0)]).get_output(0)
        
        main_body = EncoderDecoder(output_channels=self.intermediate_channels,
                                   bottleneck_block_count=self.bottleneck_block_count,
                                   kernel_ws=self.kernel_ws)
        current_index,input_tensor = main_body(network,input_tensor,self.weight_index)
        self.weight_index = current_index

        color = Conv7(network,input_tensor,self.image_channels, self.kernel_ws[self.weight_index])
        color = network.add_activation(input=color,type=trt.ActivationType.TANH).get_output(0)
        self.weight_index += 1
        alpha = Conv7(network,input_tensor,self.image_channels,self.kernel_ws[self.weight_index])
        alpha = network.add_activation(input=alpha,type=trt.ActivationType.SIGMOID).get_output(0)        
        
        ones = network.add_constant((1,1,1,1),trt.Weights(np.ones((1,1,1,1),
                                     dtype=np.float32))).get_output(0)
        
        alpha_image = network.add_elementwise(alpha,input_image,trt.ElementWiseOperation.PROD).get_output(0)
        alpha_color = network.add_elementwise(ones,alpha,trt.ElementWiseOperation.SUB).get_output(0)
        alpha_color = network.add_elementwise(alpha_color,color,trt.ElementWiseOperation.PROD).get_output(0)
        
        output_image = network.add_elementwise(alpha_image,alpha_color,trt.ElementWiseOperation.SUM).get_output(0)

        output_image.name = "morphed_image"
        network.mark_output(output_image)
        
        return network, output_image, input_shape

class Rotator:
    # Network that involves two face rotator algorihms
    def __init__(self,
                 image_channels,
                 pose_size,
                 intermediate_channels,
                 bottleneck_block_count,
                 kernel_ws):
        self.image_channels = image_channels 
        self.pose_size = pose_size 
        self.intermediate_channels = intermediate_channels 
        self.bottleneck_block_count = bottleneck_block_count
        self.kernel_ws = kernel_ws
        self.weight_index = 0
    
    def __call__(self, network, morphed_image, input_shape):
        base_grid = network.add_input(name="base_grid",dtype=trt.float32,shape=(-1,-1,-1,2))
        rotate_params = network.add_input(name="rotate_params",dtype=trt.float32,shape=(-1,self.pose_size))
        
        shuffle_layer = network.add_shuffle(rotate_params)
        shuffle_layer.reshape_dims = (0,0,1,1)
        
        slice_layer = network.add_slice(shuffle_layer.get_output(0),(0,0,0,0),(0,0,0,0),(1,1,0,0))
        slice_layer.set_input(2,input_shape)
        
        x = network.add_concatenation([morphed_image, slice_layer.get_output(0)]).get_output(0)

        main_body = EncoderDecoder(output_channels=self.intermediate_channels,
                                   bottleneck_block_count=self.bottleneck_block_count,
                                   kernel_ws=self.kernel_ws)                  

        current_index,input_tensor = main_body(network,x,self.weight_index)
        self.weight_index = current_index
        color_change = Conv7(network,input_tensor,self.image_channels, self.kernel_ws[self.weight_index])
        color_change = network.add_activation(input=color_change,type=trt.ActivationType.TANH).get_output(0)
        self.weight_index += 1
        alpha_mask = Conv7(network,input_tensor,self.image_channels,self.kernel_ws[self.weight_index])
        alpha_mask = network.add_activation(input=alpha_mask,type=trt.ActivationType.SIGMOID).get_output(0)
        self.weight_index += 1        
        
        ones = network.add_constant((1,1,1,1),trt.Weights(np.ones((1,1,1,1),
                                     dtype=np.float32))).get_output(0)
        
        alpha_image = network.add_elementwise(alpha_mask,morphed_image,trt.ElementWiseOperation.PROD).get_output(0)
        alpha_color = network.add_elementwise(ones,alpha_mask,trt.ElementWiseOperation.SUB).get_output(0)
        alpha_color = network.add_elementwise(alpha_color,color_change,trt.ElementWiseOperation.PROD).get_output(0)
        color_changed = network.add_elementwise(alpha_image,alpha_color,trt.ElementWiseOperation.SUM).get_output(0)
        
        zhou_grid_change = Conv7(network,input_tensor,2,self.kernel_ws[self.weight_index])

        shape_shuffle = network.add_shuffle(zhou_grid_change)
        shape_shuffle.second_transpose = (0,2,3,1)
        shape_shuffle = network.add_shape(shape_shuffle.get_output(0))

        grid_change = network.add_shuffle(zhou_grid_change)
        grid_change.reshape_dims = (0,0,-1)
        grid_change.second_transpose = (0,2,1)
        
        grid_change = network.add_shuffle(grid_change.get_output(0))
        grid_change.set_input(1,shape_shuffle.get_output(0))

        grid = network.add_elementwise(base_grid,grid_change.get_output(0),trt.ElementWiseOperation.SUM).get_output(0)
        resampled = GridSample(network,morphed_image,grid,0,1,1).get_output(0)
        
        color_changed.name = "color_changed"
        resampled.name = "resampled"
        
        network.mark_output(color_changed)
        network.mark_output(resampled)

        return network, color_changed, resampled

class Combiner:
    # Network that combines the results from the two rotation algorithms in the rotator network
    def __init__(self,
                 image_channels,
                 pose_size,
                 intermediate_channels,
                 bottleneck_block_count,
                 kernel_ws):
        self.image_channels = image_channels 
        self.pose_size = pose_size
        self.intermediate_channels = intermediate_channels 
        self.bottleneck_block_count = bottleneck_block_count
        self.kernel_ws = kernel_ws
        self.weight_index = 0

    def __call__(self, network, first_image, second_image, input_shape):
        combine_params = network.add_input(name="combine_params",dtype=trt.float32,shape=(-1,self.pose_size))
        
        shuffle_layer = network.add_shuffle(combine_params)
        shuffle_layer.reshape_dims = (0,0,1,1)
        
        slice_layer = network.add_slice(shuffle_layer.get_output(0),(0,0,0,0),(0,0,0,0),(1,1,0,0))
        slice_layer.set_input(2,input_shape)
        input_tensor = network.add_concatenation([first_image, second_image, slice_layer.get_output(0)]).get_output(0)

        main_body = UNetModule(output_channels = self.intermediate_channels,
                               bottleneck_block_count = self.bottleneck_block_count,
                               kernel_ws = self.kernel_ws)
        
        current_index,U_output = main_body(network,input_tensor,self.weight_index)
        self.weight_index = current_index
        combine_alpha_mask = Conv7(network,U_output,self.image_channels,self.kernel_ws[self.weight_index])
        combine_alpha_mask = network.add_activation(input=combine_alpha_mask,type=trt.ActivationType.SIGMOID).get_output(0)
        self.weight_index += 1
        
        ones = network.add_constant((1,1,1,1),trt.Weights(np.ones((1,1,1,1),
                                     dtype=np.float32))).get_output(0)
        combined_image_0 = network.add_elementwise(combine_alpha_mask,first_image,trt.ElementWiseOperation.PROD).get_output(0)
        combined_image_1 = network.add_elementwise(ones,combine_alpha_mask,trt.ElementWiseOperation.SUB).get_output(0)
        combined_image_2 = network.add_elementwise(combined_image_1,second_image,trt.ElementWiseOperation.PROD).get_output(0)
        combined_image = network.add_elementwise(combined_image_0,combined_image_2,trt.ElementWiseOperation.SUM).get_output(0)
        
        retouch_alpha_mask = Conv7(network,U_output,self.image_channels,self.kernel_ws[self.weight_index])
        retouch_alpha_mask = network.add_activation(retouch_alpha_mask,type=trt.ActivationType.SIGMOID).get_output(0)
        self.weight_index += 1
        
        retouch_color_change = Conv7(network,U_output,self.image_channels,self.kernel_ws[self.weight_index])
        retouch_color_change = network.add_activation(retouch_color_change,type=trt.ActivationType.TANH).get_output(0)

        final_0 = network.add_elementwise(retouch_alpha_mask,combined_image,trt.ElementWiseOperation.PROD).get_output(0)
        final_1 = network.add_elementwise(ones,retouch_alpha_mask,trt.ElementWiseOperation.SUB).get_output(0)
        final_2 = network.add_elementwise(final_1,retouch_color_change,trt.ElementWiseOperation.PROD).get_output(0)
        final = network.add_elementwise(final_0,final_2,trt.ElementWiseOperation.SUM).get_output(0)
        
        final.name = "final_image"
        network.mark_output(final)

        return network

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="TensorRT API Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-x", "--onnx", required=False, help="The ONNX model file path.")
    parser.add_argument("-o", "--output", required=True, default="inference.engine", help="The serialized engine file, ex inference.engine")
    parser.add_argument("-b", "--batch_size", default=[1,4,8], action="append", help="Batch size(s) to optimize for", type=int)
    parser.add_argument("-he", "--height", default=[128,256,512], action="append", type=int,help="heights to optimize for")
    parser.add_argument("-wi", "--width", default=[128,256,512], action="append", type=int,help="widths to optimize for")
    parser.add_argument("-c", "--config", default="config/talking_head.yaml",help="The folder containing the config.yaml")
    parser.add_argument("--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-w", "--workspace_size", default=4000, help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--gpu_fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    parser.add_argument("--strict_types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    args, _ = parser.parse_known_args()
    
    with open(args.config,"rb") as f:
        configs = yaml.load(f.read())

    morpher_weights = torch.load(configs["morpher_params"]["morpher_weights"])
    rotator_weights = torch.load(configs["rotator_params"]["rotator_weights"])
    combiner_weights = torch.load(configs["combine_params"]["combine_weights"])
    
    def convert_weights(ws):
        # Parse weights as numpy array
        converted_weights = []
        for i in ws:
            np_array = ws[i].cpu().numpy()
            converted_weights.append(np_array)
        return converted_weights

    morpher_weights = convert_weights(morpher_weights)
    rotator_weights = convert_weights(rotator_weights)
    combiner_weights = convert_weights(combiner_weights)

    def create_optimization_profile(builder, inputs, batch_sizes=[1,8,16], h=[128,256,256], w=[128,256,256]): 
        # Create an optimization profile
        profile = builder.create_optimization_profile()
        for inp in inputs:
            if inp.name == "input_image":
                profile.set_shape(inp.name, min=(batch_sizes[0],inp.shape[1],h[0],w[0]),
                                            opt=(batch_sizes[1],inp.shape[1],h[1],w[1]),
                                            max=(batch_sizes[2],inp.shape[1],h[2],w[2]))
                continue
            if inp.name == "base_grid":
                profile.set_shape(inp.name, min=(batch_sizes[0],h[0],w[0],inp.shape[-1]),
                                            opt=(batch_sizes[1],h[1],w[1],inp.shape[-1]),
                                            max=(batch_sizes[2],h[2],w[2],inp.shape[-1]))
                continue

            profile.set_shape(inp.name, min=(batch_sizes[0],inp.shape[-1]), 
                                        opt=(batch_sizes[1],inp.shape[-1]), 
                                        max=(batch_sizes[2],inp.shape[-1]))
        return profile
    
    logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    def add_profile(config, inputs, profile):
        # Check and add profile to builder config
        logger.debug("=== Optimization Profile ===")
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - Profile {} - Min {} Opt {} Max {}".format(inp.name, 0, _min, _opt, _max))
        config.add_optimization_profile(profile)
   
    TRT_LOGGER = trt.Logger()
    # Adjust logging verbosity
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    # Load plugins
    ctypes.CDLL("./gridSamplerPlugin/libgridsampler.so", mode = ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
   
    # Network Flag
    network_flags = 0
    network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder_flag_map = {
        'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
        'refittable': trt.BuilderFlag.REFIT,
        'debug': trt.BuilderFlag.DEBUG,
        'strict_types': trt.BuilderFlag.STRICT_TYPES,
        'fp16': trt.BuilderFlag.FP16,
        'int8': trt.BuilderFlag.INT8,
    }
    
    # Initialize the builder and start building the engine
    with trt.Builder(TRT_LOGGER) as builder,\
         builder.create_network(network_flags) as network,\
         builder.create_builder_config() as config:

         config.max_workspace_size = args.workspace_size

         # Set Builder Config Flags
         for flag in builder_flag_map:
             if getattr(args, flag):
                 logger.info("Setting {}".format(builder_flag_map[flag]))
                 config.set_flag(builder_flag_map[flag])

         if args.fp16 and not builder.platform_has_fast_fp16:
             logger.warning("FP16 not supported on this platform.")

         if args.int8 and not builder.platform_has_fast_int8:
             logger.warning("INT8 not supported on this platform.")

         if args.int8:
             raise NotImplementedError

         morpher = Morpher(image_channels=configs["morpher_params"]["image_channels"],
                           pose_size=configs["morpher_params"]["pose_size"],
                           intermediate_channels=configs["morpher_params"]["intermediate_channels"],
                           bottleneck_block_count=configs["morpher_params"]["bottleneck_block_count"],
                           kernel_ws=morpher_weights)
         morpher_network, output_image, input_shape = morpher(network)

         rotator = Rotator(image_channels=configs["rotator_params"]["image_channels"],
                           pose_size=configs["rotator_params"]["pose_size"],
                           intermediate_channels=configs["rotator_params"]["intermediate_channels"],
                           bottleneck_block_count=configs["rotator_params"]["bottleneck_block_count"],
                           kernel_ws=rotator_weights)
         rotator_network, color_changed, resampled= rotator(morpher_network,output_image,input_shape)

         combiner = Combiner(image_channels=configs["combine_params"]["image_channels"],
                             pose_size=configs["combine_params"]["pose_size"],
                             intermediate_channels=configs["combine_params"]["intermediate_channels"],
                             bottleneck_block_count=configs["combine_params"]["bottleneck_block_count"],
                             kernel_ws=combiner_weights)
         final_network = combiner(rotator_network,color_changed,resampled,input_shape)

         print("Network has ", network.num_layers, " layers")
         TRT_LOGGER.log(trt.Logger.INFO, msg="Network populated, now build the engine")

         inputs = [network.get_input(i) for i in range(network.num_inputs)]
         opt_profile = create_optimization_profile(builder, inputs, args.batch_size, args.height, args.width)
         add_profile(config, inputs, opt_profile)

         with builder.build_engine(final_network,config) as engine, \
             open(args.output,"wb") as f:
             print("serializing the engine")
             f.write(engine.serialize())
