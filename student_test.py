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
    return A.mm(B)


def torch_conv2d(inputs, weight, bias, shape):
    '''Interface of torch.nn.functional.conv2d

    Args:
    -----------------------------
    inputs, weight, bias : torch.tensor
        first three args for torch.nn.functional.conv2d

    shape   : list
        last args for torch.nn.functional.conv2d
    -----------------------------

    Returns:
    -----------------------------

    torch.tensor
    -----------------------------
    '''
    return torch.nn.functional.conv2d(inputs, weight, bias, *shape[9:])


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


def gemm(height, width, length, transposeA=False, transposeB=False, dtype="float32"):
    """Matrix multiplies matrix

    Args:
    -----------------------------
    height, width, length : int
        shape of A and B
            A: tvm.tensor.Tensor
                shape [height, width]
            B: tvm.tensor.Tensor
                shape [width, length]

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
    A = tvm.placeholder((height, width), dtype=dtype, name="A")
    B = tvm.placeholder((width, length), dtype=dtype, name="B")
    if transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[0].value == B.shape[1].value)
        C = tvm.compute((A.shape[1], B.shape[0]), lambda i,
                        j: tvm.sum(A[k, i] * B[j, k], axis=k))
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[0].value == B.shape[0].value)
        C = tvm.compute((A.shape[1], B.shape[1]), lambda i,
                        j: tvm.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, B.shape[1]))
        assert_print(A.shape[1].value == B.shape[1].value)
        C = tvm.compute((A.shape[0], B.shape[0]), lambda i,
                        j: tvm.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.reduce_axis((0, B.shape[0]))
        assert_print(A.shape[1].value == B.shape[0].value)
        C = tvm.compute((A.shape[0], B.shape[1]), lambda i,
                        j: tvm.sum(A[i, k] * B[k, j], axis=k))

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


def build_and_run(s, Tensor, control_f, shape, time_count, count=200, device_id=0, tar="llvm"):
    """ Build and record the time of running.

        Args:
        -----------------------------
        s: schedule.Schedule get form the student's auto_schedule

        Tensor  (list)
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
    except:
        print("Can not found device !!!")
        time_count.put([-1, -1])
        return -1
    # Build function form s and Tensor.
    try:
        f = tvm.build(s, Tensor, name="my_op")
    except:
        traceback.print_exc()
        print("Can not build successfully !!!")
        time_count.put([-1, -1])
        return -1
    try:
        Output_tensor = Tensor[-1]
        del Tensor[-1]
    except:
        print("The input is not correct !!!")
        time_count.put([-1, -1])
        return -1
    # Craft input data.
    try:
        Input_tvm_batch = []
        Input_torch_batch = []
        for it in range(0, count):
            Input_tvm_data = []
            Input_torch_data = []

            for i in Tensor:
                data = np.random.random(
                    [int(j) for j in i.shape]).astype(np.float32) * 100
                tvm_data = tvm.nd.array(data, ctx)
                torch_data = torch.tensor(data)
                Input_tvm_data.append(tvm_data)
                Input_torch_data.append(torch_data)

            Output_holder = tvm.nd.array(
                np.zeros([int(j) for j in Output_tensor.shape],
                         dtype=Output_tensor.dtype), ctx
            )

            Input_tvm_batch.append(Input_tvm_data + [Output_holder])
            Input_torch_batch.append(Input_torch_data)
    except:
        traceback.print_exc()
        print("Can not create input datas !!!")
        time_count.put([-1, -1])
        return -1
    control_f(*(Input_torch_batch[0]+[shape]))
    try:
        tvm_time = 0
        torch_time = 0
        for i in range(0, count):
            begin = time.time()
            f(*Input_tvm_batch[i])
            end = time.time()
            tvm_time += (end - begin)
            begin = time.time()
            control_f(*(Input_torch_batch[0]+[shape]))
            end = time.time()
            torch_time += (end - begin)
    except TimeoutError:
        print("Results of shape", shape, "Timeout!")
    except:
        print("Results of shape", shape, "\n| The culation is not correct !!!")
    print("Results of shape", shape, " \n| your time:", tvm_time, " s| pytorch time:", torch_time, "s\n")
    time_count.put([tvm_time, torch_time])


def _auto_schedule(auto_schedule_func, func, shape):
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
    return auto_schedule_func(func, shape)


def _evaluate(torch_func, func, shape, target, dev_id, times, timeout_create, timeout_cal, time_count):
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
        None
        -----------------------------
        '''

    # testing if auto_schedule work with time limit
    timelimit=ceil(timeout_create)
    signal.signal(signal.SIGALRM, handler)
    time_now=time.time()
    signal.alarm(timelimit)

    # import student module
    try:
        student_module = load_module('student_module', *find_module('auto_schedule'))
    except ImportError as e:
        print('An error occurs when importing auto_schedule module')
        print(e)
        return -1

    try:
        s, bufs = _auto_schedule(student_module.auto_schedule, func, shape)
    except Exception as e:
        traceback.print_exc()
        print("failed in auto_schedule!")
        print(e)
        return -1

    timepass=time.time()-time_now
    signal.signal(signal.SIGALRM,signal.SIG_IGN)
    if timepass>timeout_create:
        print("Timeout in auto_schedule!")
        return -1

    # testing calculating speed in Build_and_Run with time limit
    timelimit=ceil(timeout_cal)
    signal.alarm(timelimit)
    signal.signal(signal.SIGALRM, handler)
    try:
        build_and_run(s, bufs, torch_func, shape, time_count,
                times, dev_id, target)
    except Exception as e:
        print("failed in Build_and_Run!")
        print(e)
        return -1

    timepass=time.time()-time_now
    signal.signal(signal.SIGALRM,signal.SIG_IGN)
    if timepass>timeout_cal:
        # print("Timeout in Build_and_Run!")
        return -1


def evaluate(torch_func, func, shape, target, dev_id=0, timeout_create=10.0, timeout_cal=10.0, times=10, max_proc_num = 4):
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

    # proc = []
    # number = len(shape)
    # time_count = [Queue() for i in range(number)]
    time_count = Queue()
    try:
        p = Process(target=_evaluate, args=(
            torch_func, func, shape, target, dev_id, times, timeout_create,
            timeout_cal, time_count))
        p.start()
        # proc.append(p)
    except:
        print("failed in creating process!")

    # waiting for testing
    timeout = timeout_create+timeout_cal
    beg = time.time()
    try:
        while time.time() - beg < timeout:
            if p.is_alive():
                time.sleep(.1)
            else:
                break
            '''
            if any(p.is_alive() for p in proc):
                time.sleep(.1)
            else:
                break
            '''
        else:
            p.terminate()
            p.join()
    except:
        print("failed in waiting for evaluating")

    # collecting testing result
    # ans = [[[-1] for col in range(2)] for i in range(number)]
    ans = [-1, -1]
    if not time_count.empty():
        auto_time, torch_time = time_count.get()
        if torch_time == -1:
            print("time out in torch fuction!")
        elif auto_time != -1:
            ans = [auto_time, torch_time]
        else:
            print("failed in auto_schedule or time out!")

    return ans

# overload Pool in order to non-daemonize
class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NewPool(pool.Pool):
    Process = NonDaemonProcess

def parallel_evaluate():
    """evaluate process

    student level : synchro
    operator level : synchro
    shape level : asynchro
    """
    # dir preparation
    res_file = 'project2_score.txt'
    res_path = res_file

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
    def pool_evaluate(shapes, veri_func, func, target):
        # create process Pool for shapes
        p = NewPool()
        run_time = []
        # exception_stat = [0, 0, 0, 0, 0]
        exception_stat = 0
        sub_procs = []
        for shape in shapes:
            subp = p.apply_async(evaluate, (veri_func, func, shape, target))
            sub_procs.append(subp)
            '''
            if case_time <= -1:
                exception_stat[-1 - case_time] += 1
                run_time.append([1, 0])
            else:
                run_time.append(case_time)
            '''
        p.close()
        p.join()
        
        for subp in sub_procs:
            case_time = subp.get()
            if case_time[0] == -1:
                exception_stat += 1
                run_time.append([1, 0])
            else:
                run_time.append(case_time)
        score_list = list(map(score_calculate, run_time))

        return score_list, exception_stat

    # stdout logs
    logs = '\rprocessing... | [passed: {}s]'.format(int(time.time() - start_time))
    sys.stdout.write(logs + '\n')
    sys.stdout.flush()

    # evaluate
    gemm_scores, gemm_exc = pool_evaluate(gemm_shapes, torch_gemm, gemm, target)
    conv_scores, conv_exc = pool_evaluate(conv2d_shapes, torch_conv2d, conv2d_nchw, target)

    if gemm_exc + conv_exc:
        exception_info = ' exception raises in {} cases'.format(gemm_exc + conv_exc)
    else:
        exception_info = ' No exceptions'
    '''
    for i in range(5):
        if gemm_exception[i] + conv_exception[i] > 0:
            exception_info += prob_exceptions[i] + 'in {} cases'.format(gemm_exception[i] + conv_exception[i])
    '''
    score_list = gemm_scores + conv_scores

    write_score(res_path, score_list, score_item, exception_info)

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

    if time_tvm == -1:
        return -1
    perf_rate = time_torch / time_tvm
    if perf_rate <= 0.1:
        return 0
    elif 0.1 < perf_rate <= 0.2:
        return 0.5
    elif 0.2 < perf_rate <= 0.4:
        return 1.5
    elif 0.4 < perf_rate <= 0.5:
        return  2.5
    elif 0.5 < perf_rate <= 0.7:
        return 4.0
    else:
        return 5.0

if __name__ == '__main__':
    parallel_evaluate()