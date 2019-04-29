import tvm
import time
import torch
import traceback
import numpy as np
import signal
from math import ceil
from multiprocessing import Process, Queue, Pipe

# additional modules
from multiprocessing import Pool
import multiprocessing.pool as pool
import os
import sys
import shutil

from imp import find_module, load_module

# remove the auto_schedule func

def handler(signum, frame):
    raise TimeoutError()



def assert_print(a, b="Error!"):
    if a == False:
        print(b)


def torch_batch_gemm(A, B, *arg):
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
    return torch.bmm(A, B)


def torch_conv2d(inputs, weight, *arg):
    '''Interface of torch.nn.functional.conv2d

    Args:
    -----------------------------
    inputs, weight: torch.tensor

    arg     : tuple
        (bias, shape) if if_bias=True
        (shape) otherwise 
    -----------------------------

    Returns:
    -----------------------------

    torch.tensor
    -----------------------------
    '''
    if len(arg)==2:
        return torch.nn.functional.conv2d(inputs, weight, arg[0], *arg[1][9:])
    else:
        return torch.nn.functional.conv2d(inputs, weight, None, *arg[0][9:])


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
    """Batched matrix multiplies matrix

    Args:
    -----------------------------
    height, width, length : int
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


def build_and_run(s, Tensor, control_f, shape, time_count, timeout_build, timeout_cal, count=20, device_id=0, tar="llvm"):
    """ Build and record the time of running.

        Args:
        -----------------------------
        s           : schedule.Schedule get form the student's auto_schedule

        Tensor      : (list)
        the input tensors and the output tensor

        control_f   : the torch function

        shape       : arg for control_f

        time_count  : used for record the running time

        timeout_build:time limit for building
        
        timeout_cal : time limit for culation

        count       : the number rounds repeat testing

        device_id   : the id of CPU
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
        timelimit=ceil(timeout_build)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timelimit)
        begin = time.time()
        f = tvm.build(s, Tensor, name="my_op")
        timepass = time.time()-begin
        signal.signal(signal.SIGALRM,signal.SIG_IGN)
        if timepass>timeout_build:
            print("Timeout in building!")
            return -1
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

    try:
        f(*Input_tvm_batch[0])
        timelimit = ceil(timeout_cal)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timelimit)
        begin = time.time()
        for i in range(0, count):
            f(*Input_tvm_batch[i])
        tvm_time=time.time()-begin
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        if tvm_time>timeout_cal:
            print("Results of shape", shape, "Timeout!")
            tvm_time = -1
        else:
            tvm_time/=count
    except TimeoutError:
        tvm_time = -1
        print("Results of shape", shape, "Timeout!")
    except:
        tvm_time = -1
        print("Results of shape", shape, "\n| The culation is not correct !!!")

    try:
        control_f(*(Input_torch_batch[0]+[shape]))
        begin = time.time()
        for i in range(0, count):
            control_f(*(Input_torch_batch[i]+[shape]))
        torch_time=time.time()-begin
        torch_time/=count
    except TimeoutError:
        torch_time = -1
        print("Results of shape", shape, "Timeout!")
    except:
        torch_time = -1
        print("Results of shape", shape, "\n| The culation is not correct !!!")

    print("Results of shape", shape, " \n| your time:", tvm_time, " s| pytorch time:", torch_time, "s\n")
    time_count.put([tvm_time, torch_time])


def _auto_schedule(auto_schedule_func, func, shape, timeout_create):
    '''Interface of auto_schedule

        Args:
        -----------------------------
        auto_schedule_func  : auto_schedule function

        func                : conv2d_nchw or gemm

        shape               : args for auto_schedule

        timeout_create      : time limit for auto_schedule_func
        -----------------------------

        Returns:
        -----------------------------
        list:[tvm.tensor.Tensor.op] 

        list of bufs in func

        timepass: float
        -----------------------------
        '''

    timelimit=ceil(timeout_create)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timelimit)
    time_now=time.time()
    s, bufs = auto_schedule_func(func, shape)
    timepass = time.time()-time_now
    signal.signal(signal.SIGALRM,signal.SIG_IGN)
    return s, bufs, timepass


def _evaluate(torch_func, func, shape, target, dev_id, times, timeout_create, timeout_build, timeout_cal, time_count):
    '''evaluating auto_schedule in special shape

        Args:
        -----------------------------
        torch_func      :  torch_conv2d or torch_batch_gemm
            interface of torch function

        auto_schedule   : function from student 

        func            : conv2d_nchw or batch_gemm

        shape           : list
            args for func

        target          : string

        dev_id          : int

        times           : int
            times of calculating in Build_and_Run

        timeout_create  : float
            time limit in creating schedule

        timeout_build   : float
            time limit in building

        timeout_cal     : float
            time limit in calculating

        time_count      : Queue
            for testing result transferring
        -----------------------------

        Returns:
        -----------------------------
        None
        -----------------------------
        '''

    
    # import student module
    try:
        student_module = load_module('student_module', *find_module(student_id, ['../extract']))
    except ImportError:
        score_list = [0 for i in range(20)]
        write_score(student_id, res_path, score_list, score_item, 'Error in Importing')
        print('An error occurs when importing the file as module:', unpack_path)
        return -1

    # testing if auto_schedule work with time limit
    try:
        s, bufs, timepass = _auto_schedule(student_module.auto_schedule, func, shape, timeout_create)
    except:
        traceback.print_exc()
        print("failed in auto_schedule!")
        return -1

    if timepass>timeout_create:
        print("timeout in auto_schedule!")
        return -1

    # testing calculating speed in Build_and_Run with time limit
    try:
        build_and_run(s, bufs, torch_func, shape, time_count, timeout_build,
                timeout_cal, times, dev_id, target)
    except Exception as e:
        print("failed in build_and_run!")
        print(e)
        return -1


def evaluate(torch_func, student_id, func, shape, target, dev_id=0, timeout_create=10.0, timeout_build=10.0,  timeout_cal=10.0, times=10):
    '''evaluating auto_schedule with a single shape

        Args:
        -----------------------------
        torch_func      :  torch_conv2d or torch_batch_gemm
            interface of torch function

        student_id      : student module name

        func            : conv2d_nchw or batch_gemm

        shape           : a single shape
            args for func

        target          : string

        dev_id          : (optional: 0) int

        timeout_create  : (optional: 10.0) float
            time limit in creating schedule

        timeout_cal     : (optional: 10.0) float
            time limit in calculating

        timeout_build   : (optional: 10.0) float
            time limit in building

        times           : (optional: 10) int
            times of calculating in Build_and_Run
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
            torch_func, student_id, func, shape, target, dev_id, times, timeout_create, timeout_build,
            timeout_cal, time_count))
        p.start()
        # proc.append(p)
    except:
        print("failed in creating process!")

    # waiting for testing
    timeout = timeout_create+timeout_cal+timeout_build
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
    sub_dir = '../submits'
    res_dir = '../results'
    imp_dir = '../extract'
    res_file = 'project2_score.txt'
    res_path = os.path.join(res_dir, res_file)
    score_item = ['gemm' + str(i) for i in range(10)] + ['conv2d' + str(i) for i in range(10)]

    if os.path.exists(res_dir) and os.path.isdir(res_dir) and os.listdir(res_dir):
        print(res_dir, "is not empty, you'd better copy the contents and clean it, now exit...")
        return
    if not os.path.exists(res_dir) or not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    if os.path.exists(imp_dir) and os.path.isdir(imp_dir) and os.listdir(imp_dir):
        print(imp_dir, "is not empty. Automatically clean it, now continue...")
        shutil.rmtree(imp_dir, ignore_errors=True)
    if not os.path.exists(imp_dir) or not os.path.isdir(imp_dir):
        os.mkdir(imp_dir)

    total_tasks = list(os.listdir(sub_dir))

    # test coeffs; currently random
    conv2d_shapes = [[4, 6, 7, 7, 9, 2, 3, 3, 1, 2, 1, 2, 3] for i in range(10)]
    gemm_shapes = [[4, 6, 7] for i in range(10)]
    target = 'llvm'

    # for stdout logs
    count_task = 0
    num_tasks = len(total_tasks)
    start_time = time.time()

    # exception info
    # prob_exceptions = ('Import Failure', 'illegal auto_schedule', 'TLE auto_schedule', 'Build Failure', 'TLE run')

    # evaluate func
    def pool_evaluate(shapes, veri_func, student_id, func, target):
        # create process Pool for shapes
        p = NewPool()
        run_time = []
        # exception_stat = [0, 0, 0, 0, 0]
        exception_stat = 0
        sub_procs = []
        for shape in shapes:
            subp = p.apply_async(evaluate, (veri_func, student_id, func, shape, target))
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

    for filezip in total_tasks:
        # stdout logs
        count_task += 1
        logs = '\rprocessing:{} | [finished/total] = [{}/{}] | [passed: {}s]'.format(
            filezip, count_task, num_tasks, int(time.time() - start_time))
        sys.stdout.write(logs + '\n')
        sys.stdout.flush()

        # parse the packed archive
        zip_path = os.path.join(sub_dir, filezip)
        student_id = filezip.split('.')[0]
        unpack_path = os.path.join(imp_dir, student_id + '/')
        try:
            shutil.unpack_archive(zip_path, unpack_path)
        except (ValueError, ReadError):
            score_list = [0 for i in range(20)]
            write_score(student_id, res_path, score_list, score_item, 'Error in Unpacking')
            print('An error occurs when unpacking the archive:', filezip)
            continue

        # evaluate
        gemm_scores, gemm_exc = pool_evaluate(gemm_shapes, torch_batch_gemm, student_id, batch_gemm, target)
        conv_scores, conv_exc = pool_evaluate(conv2d_shapes, torch_conv2d, student_id, conv2d_nchw, target)

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

        write_score(student_id, res_path, score_list, score_item, exception_info)

    return

def write_score(student_id, res_file, score_list, score_item, prob_error=''):
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
    line = '{}: '.format(student_id)
    for i in range(len(score_item)):
        line += '{}:{} '.format(score_item[i], score_list[i])
    line += 'total:{} '.format(total_score)
    line += 'exceptions:{}\n'.format(prob_error)
    with open(res_file, 'a') as f:
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
