import os
import tensorrt as trt
import torch.nn.functional as F 
import torch
import numpy as np 
import argparse
import pycuda.autoinit
import pycuda.driver as cuda
import imageio
import ctypes
from tqdm import tqdm
from PIL import Image
import PIL.Image


def rgba_to_numpy_image(output_image):
    height = output_image.shape[1]
    width = output_image.shape[2]

    numpy_image = (output_image.reshape(4, height * width).transpose().reshape(height, width, 4) + 1.0) * 0.5
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgba_image = np.concatenate((rgb_image, a_image.reshape(height, width, 1)), axis=2)
    return rgba_image

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def extract_numpy_image_from_filelike(file):
    # preprocessing function for image file
    image = imageio.imread(file)
    image_size = image.shape[1]
    image = (image / 255.0).reshape(image_size, image_size, 4)
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    image = image.reshape((image_size * image_size, 4)).T
    image = image.reshape((4, image_size, image_size))
    image = image * 2.0 - 1.0
    image = np.ascontiguousarray(image)
    image = np.expand_dims(image,axis=0)
    image = np.array(image,dtype=np.float32)
    return image

def gen_params(num_params, random_seed=1):
    # Generate a set of random pose parameters
    np.random.seed(random_seed)
    morpher_params = []
    rotator_params = []
    
    for i in range(num_params):
        morpher = np.random.uniform(0.,1.,(1,3))
        rotator = np.random.uniform(-1.,1.,(1,3))
        morpher = np.ascontiguousarray(morpher.astype(np.float32))
        rotator = np.ascontiguousarray(rotator.astype(np.float32))
        morpher_params.append(morpher)
        rotator_params.append(rotator)
    return morpher_params, rotator_params    

def get_binding_idxs(engine, profile_index):
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    print("Engine/Binding Metadata")
    print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
    print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
    print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
    print("\tLast binding for profile {}: {}".format(profile_index, end_binding-1))
    # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)
    return input_binding_idxs, output_binding_idxs

def setup_binding_shapes(engine,context,host_inputs,input_binding_idxs,output_binding_idxs):
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)
    assert context.all_binding_shapes_specified

    host_outputs = []
    device_outputs = []
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
        buffer = cuda.pagelocked_empty(trt.volume(output_shape),np.float32)
        # Allocate output buffers on device
        device_outputs.append(cuda.mem_alloc(buffer.nbytes))
        # Allocate buffers to hold output results after copying back to host
        host_outputs.append(buffer)

    return host_outputs, device_outputs

def main():
    parser = argparse.ArgumentParser(description="running inference ")
    parser.add_argument("-e","--engine", type=str, help="location of runtime engine")
    parser.add_argument("-n","--num_params", default=300,type=int, help="number of transformation parameters")
    parser.add_argument("-o","--output", default="results/output_0.gif",help="The serialized engine file, ex inference.engine")
    parser.add_argument("-s","--source_image",required=True, help="The source image to animate")
    parser.add_argument("-f","--fps",default=10,help="fps parameter for the generated gif",type=int)
    parser.add_argument("--store_frames",action='store_true',help="store generated PNGs instead of generating a gif")
    parser.add_argument("-si","--size",default=[256,256],action="append",type=int,help="spatial size of input image")
    args = parser.parse_args()
    
    frames = []
    source_image = extract_numpy_image_from_filelike(args.source_image)

    morpher_params, rotator_params = gen_params(args.num_params)

    identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(1, 1, 1)
    base_grid = F.affine_grid(identity, [1, 4, args.size[-2], args.size[-1]], align_corners=True)
    base_grid = np.array(base_grid,dtype=np.float32)
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    #Load Custom plugin libraries
    ctypes.CDLL("./gridSamplerPlugin/libgridsampler.so", mode = ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    
    with open(args.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        print("Loaded engine:{}".format(args.engine))

        context = engine.create_execution_context()
        context.debug_sync = True

        context.active_optimization_profile = 0
        print("Active Optimization Profile:{}".format(context.active_optimization_profile))

        input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, context.active_optimization_profile)
        input_names = [engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]
        output_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]
        # Allocate device memory for inputs. This can be easily re-used if the
        # input shapes don't change
        host_inputs = [source_image,morpher_params[0],base_grid,rotator_params[0],rotator_params[0]]
        host_inputs_buffers = [cuda.pagelocked_empty_like(i) for i in host_inputs] 
        [np.copyto(i,j) for i,j in zip(host_inputs_buffers,host_inputs)]
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]

        input_binding_idxs, output_binding_idxs = get_binding_idxs(engine, context.active_optimization_profile)
        input_names = [engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]
        output_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]
        
        for h_input, d_input in zip(host_inputs_buffers, device_inputs):
            cuda.memcpy_htod(d_input, h_input)
        host_outputs, device_outputs = setup_binding_shapes(engine, context, host_inputs, input_binding_idxs, output_binding_idxs)
        print("Input Metadata")
        print("\tNumber of Inputs: {}".format(len(input_binding_idxs)))
        print("\tInput Bindings for Profile {}: {}".format(context.active_optimization_profile, input_binding_idxs))
        print("\tInput names: {}".format(input_names))
        print("\tInput shapes: {}".format([inp.shape for inp in host_inputs]))

        print("Output Metadata")
        print("\tNumber of Outputs: {}".format(len(output_binding_idxs)))
        print("\tOutput names: {}".format(output_names))
        print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
        print("\tOutput Bindings for Profile {}: {}".format(context.active_optimization_profile, output_binding_idxs))
        stream = cuda.Stream()
        for i in tqdm(range(args.num_params)):
            if i > 0 :
                host_outputs, device_outputs = setup_binding_shapes(engine, context, host_inputs, input_binding_idxs, output_binding_idxs)
                [np.copyto(host_inputs_buffers[1],morpher_params[i]),
                 np.copyto(host_inputs_buffers[3],rotator_params[i]),
                 np.copyto(host_inputs_buffers[4],rotator_params[i])]

                cuda.memcpy_htod_async(device_inputs[1],host_inputs_buffers[1],stream)
                cuda.memcpy_htod_async(device_inputs[3],host_inputs_buffers[3],stream)
                cuda.memcpy_htod_async(device_inputs[4],host_inputs_buffers[4],stream)

            # Bindings are a list of device pointers for inputs and outputs
            bindings = device_inputs + device_outputs
            bindings=[int(binding) for binding in bindings]

            # Inference
            context.execute_async_v2(bindings,stream.handle)

            # Copy outputs back to host to view results
            cuda.memcpy_dtoh_async(host_outputs[-1],device_outputs[-1],stream)
            stream.synchronize()
            # View outputs
            combined = np.reshape(host_outputs[-1],(4,args.size[-2],args.size[-1]))
            frames.append(combined)

        del context
        del engine
        print("\n Generating GIF ....")
        final_frames = [rgba_to_numpy_image(i) for i in frames]
        pil_images = [Image.fromarray(np.uint8(np.rint(f * 255.0)), mode='RGBA') for f in final_frames]
        pil_images[0].save(args.output,save_all=True, append_images=pil_images[1:], optimize=False, duration=200, loop=1,transparency=255,disposal=2)
        print("\n Done !")
        if args.store_frames:
            if not os.path.isdir("results/frames"):
                os.mkdir("results/frames")
            [pil_image.save("results/frames/result_{}.png".format(i)) for i,pil_image in enumerate(pil_images)]

if __name__=="__main__":
    main()