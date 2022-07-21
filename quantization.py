# import PyTorch
import torch
import numpy as np

from scipy.linalg import hadamard, inv

def add_dp_noise(input, sigma, clip_b):
    input_norm = input.norm()
    
    output = input / torch.maximum(torch.ones(1).to(input_norm.device), input_norm/clip_b)
    output = output + torch.empty(output.size()).normal_(mean=0, std=sigma).to(input_norm.device)

    return output

# n_bits = 2, 4, 8
# clip_b: manually set a clipping bound on top gradients, i.e., 0.01, 0.05, 0.1
# default as 0. Disable the gradient clipping
# unbiased: apply probabilistic unbiased quantization or not
def combine_quantization(input, n_bits, clip_b=0, unbiased=False):
    quanti_level = 2 ** n_bits
    sz = input.size()

    input_norm = input.norm()
    output = input / input_norm
    output = output.acos() / np.pi

    n_output = output.nelement()
    output = output.reshape( n_output )
    bound = 0.5 - (output - 0.5).abs().sort()[0][-int(n_output * clip_b)-1]

    v_min = bound
    v_max = 1-bound

    output[output > v_max] = v_max
    output[output < v_min] = v_min
    
    output = (output - v_min) / (v_max - v_min) * (quanti_level - 1)

    output_sign = output.clone()
    output_sign[...] = 0
    output_sign[output >= 8] = 1
    output_sign[output <= 7] = 1
    output[output_sign == 0] -= 7

    if unbiased:
        output = prob_quantization(output).type(torch.cuda.ByteTensor)
    else:
        output = output.round().type(torch.cuda.ByteTensor)

    output = output.reshape(sz)

    return output, output_sign, input_norm, bound

def combine_dequantization(input, input_sign, n_bits, norm, clip_b):
    quanti_level = 2 ** n_bits

    v_min = clip_b
    v_max = 1-clip_b
    input[input_sign == 0] += 7
    output = input.type(torch.cuda.FloatTensor) * (v_max - v_min) / (quanti_level - 1) + v_min
    output = torch.cos(output * np.pi) * norm

    return output

# n_bits = 2, 4, 8
# clip_b: manually set a clipping bound on top gradients, i.e., 0.01, 0.05, 0.1
# default as 0. Disable the gradient clipping
# unbiased: apply probabilistic unbiased quantization or not
def cosine_quantization(input, n_bits, clip_b=0, unbiased=False):
    quanti_level = 2 ** n_bits
    sz = input.size()

    input_norm = input.norm()
    output = input / input_norm
    output = output.acos() / np.pi

    n_output = output.nelement()
    output = output.reshape( n_output )
    bound = 0.5 - (output - 0.5).abs().sort()[0][-int(n_output * clip_b)-1]

    v_min = bound
    v_max = 1-bound

    output[output > v_max] = v_max
    output[output < v_min] = v_min
    
    output = (output - v_min) / (v_max - v_min) * (quanti_level - 1)

    if unbiased:
        output = prob_quantization(output).type(torch.cuda.ByteTensor)
    else:
        output = output.round().type(torch.cuda.ByteTensor)

    output = output.reshape(sz)

    return output, input_norm, bound

def cosine_dequantization(input, n_bits, norm, clip_b):
    quanti_level = 2 ** n_bits

    v_min = clip_b
    v_max = 1-clip_b
    output = input.type(torch.cuda.FloatTensor) * (v_max - v_min) / (quanti_level - 1) + v_min
    output = torch.cos(output * np.pi) * norm

    return output

# n_bits = 2, 4, 8
# unbiased: apply probabilistic unbiased quantization or not
# hadamard: apply random hadamard rotation or not
def linear_quantization(input, n_bits, unbiased=True, hadamard=True):
    quanti_level = 2 ** n_bits
    rand_diag = []

    if hadamard:
        input , rand_diag = hadamard_rotation(input)

    v_max = input.max()
    v_min = input.min()        
    output = input
    output = (output - v_min) / (v_max - v_min) * (quanti_level - 1)

    if unbiased:
        output = prob_quantization(output).type(torch.cuda.ByteTensor)
    else:
        output = output.round().type(torch.cuda.ByteTensor)

    #output = output.reshape(sz)

    return output, v_min, v_max, rand_diag

def linear_dequantization(input, n_bits, v_min, v_max, rand_diag, hadamard=True):
    quanti_level = 2 ** n_bits
    output = input.type(torch.cuda.FloatTensor) * (v_max - v_min) / (quanti_level - 1) + v_min
    if hadamard:
        output = hadamard_rotation_reverse(output , rand_diag)
    
    return output

def hadamard_rotation(input):
    sz = input.size()
    sz1 = sz[0]
    sz2 = int(input.nelement() / sz1)
    dim = 2 ** np.ceil(np.log2(sz1))
    hadamard_mat = hadamard(dim)
    if hadamard_mat.shape[0] != sz1:
        hadamard_mat = hadamard_mat[:sz1, :sz1]
    hadamard_mat = torch.tensor(hadamard_mat).type(input.type() )
    
    x = input.reshape(sz1, sz2)
    diag = (torch.rand(x.size()) < 0.5).type(x.type() )
    diag = diag * 2 - 1
    x = torch.mm(hadamard_mat, x) * diag
    x = x.reshape(sz)
    return x, diag

def hadamard_rotation_reverse(input, diag):
    sz = input.size()
    sz1 = sz[0]
    sz2 = int(input.nelement() / sz1)
    dim = 2 ** np.ceil(np.log2(sz1))
    hadamard_mat_inv = hadamard(dim)
    if hadamard_mat_inv.shape[0] != sz1:
        hadamard_mat_inv = hadamard_mat_inv[:sz1, :sz1]
        hadamard_mat_inv = inv(hadamard_mat_inv)
        hadamard_mat_inv = torch.tensor(hadamard_mat_inv).type(input.type() )
    else:
        hadamard_mat_inv = torch.tensor(hadamard_mat_inv).type(input.type() ) / dim

    x = input.reshape(sz1, sz2)
    x = x * diag
    x = torch.mm(hadamard_mat_inv, x)
    x = x.reshape(sz)
    return x

def prob_quantization(input):
    x = torch.ceil(input)
    p = torch.rand(x.size()).cuda()
    x = x - (p < x - input).type(x.type())
    return x
