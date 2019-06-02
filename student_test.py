# -*- coding: utf-8 -*-
import os
import sys
import time
import traceback
import signal
import multiprocessing.pool as pool
import shutil
import tvm
import torch
import numpy as np

import config

from imp import find_module, load_module
from math import ceil
from multiprocessing import Process, Queue, Pipe
from multiprocessing import Pool


def handler(signum, frame):
    raise TimeoutError()


def assert_print(a, b="Error!"):
    if a == False:
        print(b)


def torch_gemm(A, B, *arg):
    '''Interface of gemm function in pytorch

        Args:
        -----------------------------
        A, B : torch.tensor
            args for gemm function in pytorch

        *arg : just for uniform interface
        -----------------------------

        Returns:
        -----------------------------

        torch.tensor
        -----------------------------
        '''
    return A.bmm(B)


def torch_conv2d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    '''Interface of torch.nn.functional.conv2d

    Args:
    -----------------------------
    inputs, weight, bias : torch.tensor
        first three args for torch.nn.functional.conv2d
    -----------------------------

    Returns:
    -----------------------------

    torch.tensor
    -----------------------------
    '''
    return torch.nn.functional.conv2d(inputs, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : tvm.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(
        padding, (int, tvm.expr.IntImm)) else padding
    assert_print(isinstance(padding, tuple),
                 "type(padding)={}".format(type(padding)))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert_print(len(padding) == 4)

    padding_zero = 0.0 if "float" in inputs.dtype else 0

    batch_size, in_channel, height, width = inputs.shape
    return tvm.compute(
        (batch_size, in_channel, height +
         padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.if_then_else(
            tvm.all(h >= padding[0], h < height + padding[0],
                    w >= padding[2], w < width + padding[2]),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero
        )
    )



def batch_gemm(batch, height, width, length, transposeA=False, transposeB=False, dtype="float32"):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    batch, height, width, length : int
        shape of A and B
            A: tvm.tensor.Tensor
                shape [batch, height, width]
            B: tvm.tensor.Tensor
                shape [batch, width, length]

    transposeA: (optional:False) bool

    transposeB: (optional:False) bool
    -----------------------------

    Returns:
    -----------------------------
    list [tvm.tensor.Tensor.op]

    list of bufs:
        shape [A, B, C]
    -----------------------------
    """
    A = tvm.placeholder((batch, height, width), dtype=dtype, name="A")
    B = tvm.placeholder((batch, width, length), dtype=dtype, name="B")
    if transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[1].value == B.shape[2].value)
        C = tvm.compute(
            (A.shape[0], A.shape[2], B.shape[1]), 
            lambda b, i, j: tvm.sum(A[b, k, i] * B[b, j, k], axis=k)
            )
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        C = tvm.compute(
            (A.shape[0], A.shape[2], B.shape[2]), 
            lambda b, i, j: tvm.sum(A[b, k, i] * B[b, k, j], axis=k)
            )
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[2]))
        assert_print(A.shape[2].value == B.shape[2].value)
        C = tvm.compute(
            (A.shape[0], A.shape[1], B.shape[1]), 
            lambda b, i, j: tvm.sum(A[b, i, k] * B[b, j, k], axis=k)
            )
    else:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[2].value == B.shape[1].value)
        C = tvm.compute(
            (A.shape[0], A.shape[1], B.shape[2]), 
            lambda b, i, j: tvm.sum(A[b, i, k] * B[b, k, j], axis=k)
            )

    return [C.op], [A, B, C]


def conv2d_nchw(batch_size, in_channel, inputs_height, inputs_width, out_channel, channel_per_group, kernel_height, kernel_width, if_bias=0, stride=1, padding=0, dilation=1, groups=1, dtype="float32"):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    batch_size, in_channel, inputs_height, inputs_width          : int
        shape of inputs
            inputs  : tvm.tensor.Tensor
                shape [batch, channel, height, width]

    out_channel, channel_per_group, kernel_height, kernel_width  : int
        shape of weight
            weight  : tvm.tensor.Tensor
                shape [out_channel, channel // groups, kernel_height, kernel_width]

    if_bias : (optional:0) bool
        bias  : tvm.tensor.Tensor
            shape [out_channel]

    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------

    list:[tvm.tensor.Tensor.op] 

    list of bufs:
        [inputs, weight, bias, Output] if if_bias
        [inputs, weight, Output] otherwise
    -----------------------------
    """
    in_h, in_w, k_h, k_w = inputs_height, inputs_width, kernel_height, kernel_width
    inputs = tvm.placeholder(
        (batch_size, in_channel, in_h, in_w), dtype=dtype, name="inputs")
    weight = tvm.placeholder(
        (out_channel, channel_per_group, k_h, k_w), dtype=dtype, name="weight")
    if if_bias:
        bias = tvm.placeholder((out_channel,), dtype=dtype, name="bias")
    assert_print(channel_per_group * groups == in_channel)
    out_channel_per_group = out_channel // groups
    assert_print(out_channel_per_group * groups == out_channel)

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.expr.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.expr.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.expr.IntImm)) else dilation
    assert_print(isinstance(stride, tuple) and len(stride) == 2)
    assert_print(isinstance(padding, tuple) and len(padding) == 2)
    assert_print(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.reduce_axis((0, channel_per_group))
    rh = tvm.reduce_axis((0, k_h))
    rw = tvm.reduce_axis((0, k_w))

    padded = zero_pad2d(inputs, padding=padding)
    Output = tvm.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc,
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
             * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if if_bias:
        Output = tvm.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: Output[b, c, h, w] + bias[c]
        )
        return [Output.op], [inputs, weight, bias, Output]
    return [Output.op], [inputs, weight, Output]


def build_and_run(s, tensors, control_f, shape, time_count, count=10, device_id=0, target="llvm", timeout=10.0):
    """ Build and record the time of running.

        Args:
        -----------------------------
        s: schedule.Schedule get form the student's auto_schedule

        tensors  (list)
        the input tensors and the output tensor

        control_f  the torch function

        shape 

        time_count: used for record the running time

        count: the number rounds repeat testing

        device_id : the id of CPU
        -----------------------------

        Returns:
        -----------------------------
        [tvm_time, torch_time]:
            [float , flaot]
        which indicates
        the total time of running scheduled tvm calculation and
        the total time of running torch calculation
        -----------------------------
        """
    # Create ctx.
    try:
        ctx = tvm.cpu(device_id)
    except Exception as e:
        string = "Can not found device !!!\n" + str(e)
        time_count.put([string, -1])
        return -1
    
    try:
        output_tensor = tensors[-1]
        del tensors[-1]
    except Exception as e:
        string = "The input is not correct !!!" + str(e)
        time_count.put([string, -1])
        return -1
    # Craft input data.
    try:
        input_tvm = []
        input_torch = []

        for tensor in tensors:
            data = np.random.random(
                [int(j) for j in tensor.shape]).astype(np.float32) * 100
            tvm_data = tvm.nd.array(data, ctx)
            torch_data = torch.tensor(data)
            input_tvm.append(tvm_data)
            input_torch.append(torch_data)

        output_holder = tvm.nd.array(
            np.zeros([int(j) for j in output_tensor.shape],
                        dtype=output_tensor.dtype), ctx
        )

        input_tvm = input_tvm + [output_holder]
    except Exception as e:
        string = "Can't prepare input data!!!\n" + str(e)
        time_count.put([string, -1])
        return -1
    
    torch_args = []
    # TODO use shape length to distinguish conv2d and gemm is foolish
    # No bias if this is convolution
    if len(shape) > 8 and shape[8] == 0:
        torch_args.append(None)
    torch_args.extend(shape[9:])
    # warm-up
    control_f(*(input_torch + torch_args))
    begin = time.time()
    for i in range(0, count):
        control_f(*(input_torch + torch_args))
    end = time.time()
    torch_time = (end - begin) * 1e3 / count

    # Build function form s and tensors.
    try:
        func = tvm.build(s, tensors + [output_tensor], target=target)
    except Exception as e:
        string = "Can not build successfully !!!" + str(e)
        time_count.put([string, torch_time])
        return -1

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ceil(timeout))
    try:
        evaluator = func.time_evaluator(func.entry_name, ctx, number=count)
        tvm_time = evaluator(*input_tvm).mean * 1e3
    except TimeoutError:
        string = "Timeout when evaluating, the limit is {}ms".format(timeout / count * 1e3)
        time_count.put([string, torch_time])
        return -1
    except Exception as e:
        string = "The culation is not correct !!!\n" + str(e)
        time_count.put([string, torch_time])
        return -1
    finally:
        # restore the default handler
        signal.signal(signal.SIGALRM,signal.SIG_IGN)
    time_count.put([tvm_time, torch_time])
    return 0


def _auto_schedule(auto_schedule_func, func, shape, queue, timeout=20 * 60):
    '''Interface of auto_schedule

        Args:
        -----------------------------
        auto_schedule_func : auto_schedule function

        func    : conv2d_nchw or gemm

        shape   : args for auto_schedule
        -----------------------------

        Returns:
        -----------------------------
        list:[tvm.tensor.Tensor.op] 

        list of bufs in func
        -----------------------------
        '''
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ceil(timeout))
    try:
        s, bufs = auto_schedule_func(func, shape)
    except TimeoutError:
        string = "Timeout when running `auto_schedule` function, time limit is {}ms".format(timeout * 1e3)
        queue.put([string, -1])
        return None, None
    except Exception as e:
        string = "Error occurs when running `auto_schedule`\n" + str(e)
        queue.put([string, -1])
        return None, None
    finally:
        # restore the default handler
        signal.signal(signal.SIGALRM,signal.SIG_IGN)
    return s, bufs


def _evaluate(torch_func, func, shape, time_count, target="llvm", dev_id=0, times=10, timeout_create=20 * 60, timeout_cal=10.0):
    '''evaluating auto_schedule in special shape

        Args:
        -----------------------------
        torch_func  :  torch_conv2d or torch_gemm
            interface of torch function

        auto_schedule: function from student 

        func        : conv2d_nchw or gemm

        shape       : list
            args for func

        target      : string

        dev_id      : int

        times       : int
            times of calculating in Build_and_Run

        timeout_create  : (optional: 10.0) float
            time limit in creating schedule

        timeout_cal     : (optional: 10.0) float
            time limit in calculating

        time_count  : Queue
            for testing result transferring
        -----------------------------

        Returns:
        -----------------------------
        '''
    # import student module
    try:
        student_module = load_module('student_module', *find_module('auto_schedule'))
    except ImportError as e:
        string = 'An error occurs when importing `auto_schedule` module\n' + str(e)
        time_count.put([string, -1])
        return -1

    # scheduling
    s, bufs = _auto_schedule(student_module.auto_schedule, func, shape, time_count, timeout_create)
    if s is None or bufs is None:
        return -1

    # evaluating
    ret = build_and_run(s, bufs, torch_func, shape, time_count, times, dev_id, target, timeout_cal)
    if ret < 0:
        return -1
    return 0


def evaluate(torch_func, func, shape, target="llvm", dev_id=0, timeout_create=20 * 60, timeout_cal=10.0, times=10):
    '''evaluating auto_schedule with a single shape

        Args:
        -----------------------------
        torch_func      :  torch_conv2d or torch_gemm
            interface of torch function

        func            : conv2d_nchw or gemm

        shape           : a single shape
            args for func

        target          : string

        dev_id          : (optional: 0) int

        timeout_create  : (optional: 10.0) float
            time limit in creating schedule

        timeout_cal     : (optional: 10.0) float
            time limit in calculating

        times           : (optional: 10) int
            times of calculating in Build_and_Run

        max_proc_num    : (optional: 4) int
        -----------------------------

        Returns:
        -----------------------------
        list    : [auto_time,torch_time] for each shape
        -----------------------------
        '''
    assert shape != None, "empty shape!"

    time_count = Queue()
    try:
        p = Process(target=_evaluate, args=(
            torch_func, func, shape, time_count, target, dev_id, times, timeout_create,
            timeout_cal))
        p.start()
    except Exception as e:
        print("Failed in creating process for shape {}\n{}".format(shape, str(e)))

    # waiting for testing
    timeout = timeout_create + timeout_cal + 10.0 # make up for torch wastes
    beg = time.time()
    try:
        while time.time() - beg < timeout:
            if p.is_alive():
                time.sleep(.1)
            else:
                break
        else:
            p.terminate()
            p.join()
    except Exception as e:
        print("Shape {}: failed in process\n{}".format(shape, str(e)))

    # collecting testing result
    ans = [-1, -1]
    if not time_count.empty():
        auto_time, torch_time = time_count.get()
        if isinstance(auto_time, str):
            print("Exceptons occur in shape {}".format(shape))
            print(auto_time)
        else:
            ans = [auto_time, torch_time]
    else:
        print("Shape {} can't get results!".format(shape))
    # clean the queue
    time_count.close()
    del time_count
    return ans

# overload Pool in order to non-daemonize
# class NonDaemonProcess(Process):
#     def __init__(self, *args, **kwargs):
#         super(Process, self).__init__(*args, **kwargs)

#     def _get_daemon(self):
#         return False

#     def _set_daemon(self, value):
#         pass

#     daemon = property(_get_daemon, _set_daemon)

# class NewPool(pool.Pool):
#     def __init__(self, *args, **kwargs):
#         super(pool.Pool, self).__init__(*args, **kwargs)

#     Process = NonDaemonProcess

class NewPool(pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NewPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc

def parallel_evaluate(parallel=1):
    """evaluate process

    student level : synchro
    operator level : synchro
    shape level : asynchro
    """
    # dir preparation
    res_file = 'project2_score.txt'
    res_path = res_file
    time_create = 20 * 60
    time_cal = 10.0
    number_test = 10

    # test coeffs; currently random
    conv2d_shapes = config.conv_shapes.copy()
    gemm_shapes = config.gemm_shapes.copy()
    np.random.shuffle(conv2d_shapes)
    np.random.shuffle(gemm_shapes)
    score_item = ['gemm_' + str(s) for s in gemm_shapes] + ['conv2d_' + str(s) for s in conv2d_shapes]
    target = 'llvm'

    # for stdout logs
    start_time = time.time()

    # exception info
    # prob_exceptions = ('Import Failure', 'illegal auto_schedule', 'TLE auto_schedule', 'Build Failure', 'TLE run')

    # evaluate func
    def pool_evaluate(shapes, veri_func, func, target="llvm"):
        # create process Pool for shapes
        p = NewPool()
        run_time = []
        # exception_stat = [0, 0, 0, 0, 0]
        exception_stat = 0
        sub_procs = []
        for i, shape in enumerate(shapes):
            subp = p.apply_async(evaluate, (veri_func, func, shape, target, i, time_create, time_cal, number_test))
            sub_procs.append(subp)

        p.close()
        p.join()
        
        ret = []
        for i, subp in enumerate(sub_procs):
            try:
                case_time = subp.get(timeout=1.0)
            except Exception as e:
                print("Can't get shape {} results\n{}".format(shapes[i], str(e)))
                case_time = [-1, -1]
            if case_time[0] < 0:
                exception_stat += 1
            ret.append(case_time)

        return ret, exception_stat

    # stdout logs
    logs = '\rProcessing begins...'
    sys.stdout.write(logs + '\n')
    sys.stdout.flush()

    # evaluate
    num_gemms = len(gemm_shapes)
    outer = ceil(num_gemms / parallel)
    gemm_ret = []
    gemm_error_count = 0
    for i in range(outer):
        part_gemm_ret, part_gemm_error = pool_evaluate(gemm_shapes[i * parallel:(i+1) * parallel], torch_gemm, batch_gemm, target)
        gemm_ret.extend(part_gemm_ret)
        gemm_error_count += part_gemm_error

    num_convs = len(conv2d_shapes)
    outer = ceil(num_convs / parallel)
    conv_ret = []
    conv_error_count = 0
    for i in range(outer):
        part_conv_ret, part_conv_error = pool_evaluate(conv2d_shapes[i * parallel:(i+1) * parallel], torch_conv2d, conv2d_nchw, target)
        conv_ret.extend(part_conv_ret)
        conv_error_count += part_conv_error
    
    if gemm_error_count or conv_error_count:
        exception_info = ' exception raises in {} cases'.format(gemm_error_count + conv_error_count)
    else:
        exception_info = ' No exceptions'
    
    print()
    print("#####################################################")
    print("The results:\n")
    string = "Time costs of GEMMs\n"
    for shape, ret in zip(gemm_shapes, gemm_ret):
        times = [ret[0] if ret[0] > 0 else "Timeout", ret[1] if ret[1] > 0 else "Not evaluted"]
        string += "{}: yours: {}(ms), torch: {}(ms)\n".format(shape, times[0], times[1])
    print(string)

    string = "Time costs of Conv2ds\n"
    for shape, ret in zip(conv2d_shapes, conv_ret):
        times = [ret[0] if ret[0] > 0 else "Timeout", ret[1] if ret[1] > 0 else "Not evaluted"]
        string += "{}: yours: {}(ms), torch: {}(ms)\n".format(shape, times[0], times[1])
    print(string)

    score_list = list(map(score_calculate, gemm_ret + conv_ret))

    write_score(res_path, score_list, score_item, exception_info)

    # stdout logs
    logs = '\rall done!'
    sys.stdout.write(logs + '\n')
    sys.stdout.flush()
    return

def write_score(res_file, score_list, score_item, prob_error=''):
    """write score into result file

    Parameters
    ----------
    student_id: str
    res_file: str
        path of file to record scores
    score_list: list
        scores in each test
    score_item: list
        test names
    prob_error: str
        exceptions and errors occurring during tests

    Returns
    -------
    """
    total_score = sum(score_list)
    line = '{}:\n'.format("your scores")
    for i in range(len(score_item)):
        line += '{}:{}\n'.format(score_item[i], score_list[i])
    line += 'total:{}\n'.format(total_score)
    line += 'exceptions:{}\n'.format(prob_error)
    with open(res_file, 'w') as f:
        f.write(line)
    print(line)
    return


def score_calculate(time_tuple):
    """scores based on look-up table

    Parameters
    ----------
    time_tuple: list
        with format [auto_time, torch_time]

    Returns
    -------
    case_score: float
        scores calculated based on the look-up table
    """
    time_tvm = time_tuple[0]
    time_torch = time_tuple[1]

    if time_tvm < 0:
        return 0.0
    perf_rate = time_torch / time_tvm
    if 0 <= perf_rate < 0.1:
        return 0.0
    elif 0.1 <= perf_rate < 0.2:
        return 0.7
    elif 0.2 <= perf_rate < 0.3:
        return 1.4
    elif 0.3 <= perf_rate < 0.4:
        return 2.1
    elif 0.4 <= perf_rate < 0.5:
        return 2.8
    elif 0.5 <= perf_rate < 0.6:
        return 3.5
    elif 0.6 <= perf_rate < 0.7:
        return 4.2
    elif 0.7 <= perf_rate < 0.8:
        return 4.9 
    elif 0.8 <= perf_rate < 0.9:
        return 5.6
    else:
        return 7.0

if __name__ == '__main__':
    parallel_evaluate()