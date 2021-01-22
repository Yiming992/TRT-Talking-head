import os
import sys
import argparse
import torch.nn.functional as F 
from tkinter import Frame, Label, BOTH, Tk, LEFT, HORIZONTAL, Scale, Button, GROOVE, filedialog, PhotoImage, messagebox
import PIL.Image
import PIL.ImageTk
import numpy as np
import torch
import tensorrt as trt 
import pycuda.autoinit 
import pycuda.driver as cuda
from run_inference import setup_binding_shapes, get_binding_idxs, extract_numpy_image_from_filelike, linear_to_srgb, srgb_to_linear, rgba_to_numpy_image
import ctypes

class PoseParameter:
    def __init__(self,
                 name: str,
                 display_name: str,
                 lower_bound: float,
                 upper_bound: float,
                 default_value: float):
        self._name = name
        self._display_name = display_name
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._default_value = default_value

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._display_name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def default_value(self):
        return self._default_value

class ManualPoserApp:
    def __init__(self,
                 master,
                 image_size,
                 engine_file):
        super().__init__()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        ctypes.CDLL("./gridSamplerPlugin/libgridsampler.so", mode = ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)
        
        file = open(engine_file,"rb")
        self.engine = runtime.deserialize_cuda_engine(file.read())
        file.close()
        
        self.context = self.engine.create_execution_context()
        self.context.debug_sync = True 
        self.context.active_optimization_profile = 0
        
        self.image_size = (1,4,image_size[-2],image_size[-1])
        self.dummy_image = np.zeros(self.image_size,dtype=np.float32)

        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).repeat(1, 1, 1)
        base_grid = F.affine_grid(identity, [1, 4, self.image_size[-2], self.image_size[-1]], align_corners=True)
        base_grid = np.array(base_grid,dtype=np.float32)
        
        self.base_grid = base_grid

        self.input_binding_idxs, self.output_binding_idxs = get_binding_idxs(self.engine, self.context.active_optimization_profile)

        self.master = master
        self.master.title("Manual Poser")

        source_image_frame = Frame(self.master, width=self.image_size[-2], height=self.image_size[-1])
        source_image_frame.pack_propagate(0)
        source_image_frame.pack(side=LEFT)

        self.source_image_label = Label(source_image_frame, text="Nothing yet!")
        self.source_image_label.pack(fill=BOTH, expand=True)

        control_frame = Frame(self.master, borderwidth=2, relief=GROOVE)
        control_frame.pack(side=LEFT, fill='y')

        self.pose_parameters = [PoseParameter("left_eye", "Left Eye", 0.0, 1.0, 0.0),
                                PoseParameter("right_eye", "Right Eye", 0.0, 1.0, 0.0),
                                PoseParameter("mouth", "Mouth", 0.0, 1.0, 1.0),
                                PoseParameter("head_x", "Head X", -1.0, 1.0, 0.0),
                                PoseParameter("head_y", "Head Y", -1.0, 1.0, 0.0),
                                PoseParameter("neck_z", "Neck Z", -1.0, 1.0, 0.0)]

        self.param_sliders = []
        for param in self.pose_parameters:
            slider = Scale(control_frame,
                           from_=param.lower_bound,
                           to=param.upper_bound,
                           length=256,
                           resolution=0.001,
                           orient=HORIZONTAL)
            slider.set(param.default_value)
            slider.pack(fill='x')
            self.param_sliders.append(slider)

            label = Label(control_frame, text=param.display_name)
            label.pack()
        
        morpher_params = np.array([[self.pose_parameters[0].default_value,
                                    self.pose_parameters[1].default_value,
                                    self.pose_parameters[2].default_value]],dtype=np.float32)
        rotator_params = np.array([[self.pose_parameters[3].default_value,
                                    self.pose_parameters[4].default_value,
                                    self.pose_parameters[5].default_value]],dtype=np.float32)
        
        self.morpher = np.ascontiguousarray(morpher_params.astype(np.float32))
        self.rotator = np.ascontiguousarray(rotator_params.astype(np.float32))
        
        self.device_inputs = [cuda.mem_alloc(self.dummy_image.nbytes),
                              cuda.mem_alloc(self.morpher.nbytes),
                              cuda.mem_alloc(self.base_grid.nbytes),
                              cuda.mem_alloc(self.rotator.nbytes),
                              cuda.mem_alloc(self.rotator.nbytes)]
        self.host_inputs = [self.dummy_image,self.morpher,self.base_grid,self.rotator,self.rotator]
        
        self.host_outputs, self.device_outputs = setup_binding_shapes(self.engine,self.context,self.host_inputs,self.input_binding_idxs,self.output_binding_idxs)
        posed_image_frame = Frame(self.master, width=self.image_size[-2], height=self.image_size[-1])
        posed_image_frame.pack_propagate(0)
        posed_image_frame.pack(side=LEFT)

        self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
        self.posed_image_label.pack(fill=BOTH, expand=True)

        self.load_source_image_button = Button(control_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = len(self.pose_parameters)
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)

    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="images")
        if len(file_name) > 0:
            image = PhotoImage(file=file_name)
            if image.width() != self.image_size[-1] or image.height() != self.image_size[-2]:
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.width(), image.height(), self.image_size[-1], self.image_size[-2])
                messagebox.showerror("Wrong image size!", message)
            self.source_image_label.configure(image=image, text="")
            self.source_image_label.image = image
            self.source_image_label.pack()

            self.source_image = extract_numpy_image_from_filelike(file_name)
            cuda.memcpy_htod(self.device_inputs[0],self.source_image)
            cuda.memcpy_htod(self.device_inputs[2],self.base_grid)
            self.needs_update = True

    def update_pose(self):
        self.current_pose = np.zeros(self.pose_size, dtype=np.float32)
        self.current_pose = np.ascontiguousarray(self.current_pose)
        for i in range(self.pose_size):
            self.current_pose[i] = self.param_sliders[i].get()
        self.current_pose = np.expand_dims(self.current_pose, axis=0)

    def update_image(self):
        self.update_pose()
        if (not self.needs_update) and self.last_pose is not None and (
                np.sum(np.abs(self.last_pose - self.current_pose)) < 1e-5):
            self.master.after(1000 // 30, self.update_image)
            return
        if self.source_image is None:
            self.master.after(1000 // 30, self.update_image)
            return
        self.last_pose = self.current_pose
        
        self.morpher = self.current_pose[:,:3]
        self.rotator = self.current_pose[:,3:]
        self.host_inputs = [self.source_image,self.morpher,self.base_grid,self.rotator,self.rotator]

        cuda.memcpy_htod(self.device_inputs[1], self.morpher)
        cuda.memcpy_htod(self.device_inputs[3], self.rotator)
        cuda.memcpy_htod(self.device_inputs[4], self.rotator)
        self.host_outputs, self.device_outputs = setup_binding_shapes(self.engine,self.context,self.host_inputs,self.input_binding_idxs,self.output_binding_idxs)
        bindings = self.device_inputs + self.device_outputs
        bindings=[int(binding) for binding in bindings]

        self.context.execute_v2(bindings)
        cuda.memcpy_dtoh(self.host_outputs[-1],self.device_outputs[-1])
        posed_image = np.reshape(self.host_outputs[-1],(4,self.image_size[-2],self.image_size[-1]))
        numpy_image = rgba_to_numpy_image(posed_image)
        pil_image = PIL.Image.fromarray(np.uint8(np.rint(numpy_image * 255.0)), mode='RGBA')
        photo_image = PIL.ImageTk.PhotoImage(image=pil_image)

        self.posed_image_label.configure(image=photo_image, text="")
        self.posed_image_label.image = photo_image
        self.posed_image_label.pack()
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)

if __name__ == "__main__":
    # Run the app
    parser = argparse.ArgumentParser(description="Add necessary app Arguments")
    parser.add_argument("-e","--engine",default="model.engine",type=str, help="location of runtime engine")
    parser.add_argument("-s","--size",default=[256,256],action="append",type=int,help="spatial size of input image")
    args = parser.parse_args()
    root = Tk()
    app = ManualPoserApp(master=root, image_size=args.size, engine_file=args.engine)
    root.mainloop()