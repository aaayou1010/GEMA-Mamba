# the code is modified from https://github.com/d-li14/involution/blob/main/cls/mmcls/models/utils/involution_cuda.py
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template
import math

from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


# @cupy._util.memoize(for_each_device=True)
# def load_kernel(kernel_name, code, **kwargs):
#     code = Template(code).substitute(**kwargs)
#     kernel_code = cupy.cuda.compile_with_cache(code)
#     return kernel_code.get_function(kernel_name)

@cupy._util.memoize(for_each_device=True) 
def load_kernel(kernel_name, code, **kwargs): # Substitute the kwargs in the kernel code template 
	code = Template(code).substitute(**kwargs) # Create a RawKernel object directly from the code 
	kernel = cupy.RawKernel(code, kernel_name) # RawKernel object is directly callable, so you can return it 
	return kernel


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_shift_kernel = kernel_loop + '''
extern "C"
__global__ void shift_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));

    ${Dtype} value = 0;

    if (${dim} == 2){
        if ((h + s >= 0 && h + s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h + s) * ${width} + w;
             value = bottom_data[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w + s >= 0 && w + s< ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w + s;
            value = bottom_data[offset];
            }
    }

    top_data[index] = value;
  }
}
'''


_shift_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void shift_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));

    ${Dtype} value = 0;
    if (${dim} == 2){
        if ((h - s >= 0 && h - s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h - s) * ${width} + w;
             value = top_diff[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w - s >= 0 && w - s < ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w - s;
            value = top_diff[offset];
        }
    }

    bottom_diff[index] = value;
  }
}
'''


class _shift(Function):
    @staticmethod
    def forward(ctx, input, shift, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = input.new(batch_size, channels, height, width)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('shift_forward_kernel', _shift_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            shift=shift, dim=dim, group=int(math.ceil(channels/shift))
                            )

            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input)
        ctx.shift, ctx.dim = shift, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input = ctx.saved_tensors[0]
        shift, dim = ctx.shift, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width,
                   shift=shift, dim=dim, group=int(math.ceil(channels/shift))
              )

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('shift_backward_grad_input_kernel',
                                _shift_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, None, None
 
def _shift_cuda(input, shift, dim):
    """ shift kernel
    """
    assert shift >=3 and shift % 2 == 1
    assert dim == 2 or dim == 3

    if input.is_cuda:
        out = _shift.apply(input, shift, dim)
    else:
        raise NotImplementedError
    return out


class Shift(nn.Module):
    def __init__(self,
                 kernel_size,
                 dim):
        super(Shift, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        assert dim == 2 or dim == 3
        assert kernel_size % 2 == 1

    def forward(self, x):
        if self.kernel_size == 1:
            return x

        out = _shift_cuda(x, self.kernel_size, self.dim)
        return out


def torch_shift(x, shift_size, dim):
    B_, C, H, W = x.shape
    pad = shift_size // 2

    x = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(x, shift_size, 1)
    x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-pad, pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_cat = torch.narrow(x_cat, 3, pad, W)
    return x_cat



def MyNorm(dim):
    return nn.GroupNorm(1, dim)

class AxialShift(nn.Module):
    r""" Axial shift  

    Args:
        dim (int): Number of input channels.
        shift_size (int): shift size .
        as_bias (bool, optional):  If True, add a learnable bias to as mlp. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, shift_size, as_bias=True, proj_drop=0., conv_kernel=3):

        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2
        # self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        # self.conv2_1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        # self.conv2_2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        # self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=as_bias)
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, padding="same", groups=1, bias=as_bias)
        self.conv2_1 = nn.Conv2d(dim, dim, conv_kernel, 1, padding="same", groups=1, bias=as_bias)
        self.conv2_2 = nn.Conv2d(dim, dim, conv_kernel, 1, padding="same", groups=1, bias=as_bias)
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, padding="same", groups=1, bias=as_bias)

        self.actn = nn.GELU()

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)

        self.shift_dim2 = Shift(self.shift_size, 2)                                                   
        self.shift_dim3 = Shift(self.shift_size, 3)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, C, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actn(x)
       
        x_shift_lr = self.shift_dim3(x)
        x_shift_td = self.shift_dim2(x)
        
        x_lr = self.conv2_1(x_shift_lr)
        x_td = self.conv2_2(x_shift_td)

        x_lr = self.actn(x_lr)
        x_td = self.actn(x_td)

        x = x_lr + x_td
        x = self.norm2(x)

        x = self.conv3(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, shift_size={self.shift_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # conv1 
        flops += N * self.dim * self.dim
        # norm 1
        flops += N * self.dim
        # conv2_1 conv2_2
        flops += N * self.dim * self.dim * 2
        # x_lr + x_td
        flops += N * self.dim
        # norm2
        flops += N * self.dim
        # norm3
        flops += N * self.dim * self.dim
        return flops


class AxialShiftedBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        as_bias (bool, optional): If True, add a learnable bias to Axial Mlp. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size=7,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.axial_shift_1 = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop)
        self.axial_shift_2 = AxialShift(dim, shift_size=shift_size, as_bias=as_bias, proj_drop=drop, conv_kernel=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_conv(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape

        shortcut = x
        x = self.norm1(x)
        y = x.clone()

        # axial shift block
        x = self.axial_shift_1(x)  # B, C, H, W
        y = self.axial_shift_2(y)  # B, C, H, W

        x = x + y

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # shift mlp 
        flops += self.axial_shift.flops(H * W)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer_mlp(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, shift_size,
                 mlp_ratio=4., as_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_size=shift_size,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
                              drop=drop, 
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Mlp_conv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
