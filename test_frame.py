import os
import sys
import tvm
import time
import torch
import config
import signal
import shutil
import traceback
from math import ceil
from multiprocessing import Pool, Process, Queue, Pipe
from imp import find_module, load_module
import numpy as np
import multiprocessing.pool as pool



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


def torch_conv2d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    '''Interface of torch.nn.functional.conv2d

    Args:
    -----------------------------
    inputs, weight: torch.tensor

    bias  : tvm.tensor.tensor
        shape [out_channel]

    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
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


def build_and_run(s, tensors, shape, time_count, count=10, device_id=0, target="llvm", timeout=10.0):
    """ Build and record the time of running.

        Args:
        -----------------------------
        s           : schedule.Schedule get form the student's auto_schedule

        Tensor      : list
            the input tensors and the output tensor

        control_f   : the torch function

        shape       : arg for control_f

        time_count  : used for record the running time

        count       : (optional: 10)the number rounds repeat testing

        device_id   : (optional: 0)the id of CPU

        timeout     : (optional: 10.0)time limit for culation
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
        time_count.put(string)
        return -1
    # Craft input data.

    try:
        input_tvm = []

        for tensor in tensors:
            data = np.random.random(
                [int(j) for j in tensor.shape]).astype(np.float32) * 100
            tvm_data = tvm.nd.array(data, ctx)
            input_tvm.append(tvm_data)

        output_holder = tvm.nd.array(
            np.zeros([int(j) for j in output_tensor.shape],
                        dtype=output_tensor.dtype), ctx
        )

        input_tvm = input_tvm + [output_holder]
    except Exception as e:
        string = "Can't prepare input data!!!\n" + str(e)
        time_count.put(string)
        return -1

    # Build function form s and tensors.
    try:
        func = tvm.build(s, tensors + [output_tensor], target=target)
    except Exception as e:
        string = "Can not build successfully !!!" + str(e)
        time_count.put(string)
        return -1

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(ceil(timeout))
    try:
        evaluator = func.time_evaluator(func.entry_name, ctx, number=count)
        tvm_time = evaluator(*input_tvm).mean * 1e3
    except TimeoutError:
        string = "Timeout when evaluating, the limit is {}ms".format(timeout / count * 1e3)
        time_count.put(string)
        return -1
    except Exception as e:
        string = "The culation is not correct !!!\n" + str(e)
        time_count.put(string)
        return -1
    finally:
        # restore the default handler
        signal.signal(signal.SIGALRM,signal.SIG_IGN)
    time_count.put(tvm_time)
    return 0


def torch_run(func, control_f, shape, count=10):
    """ Build and record the time of running.

        Args:
        -----------------------------
        s           : schedule.Schedule get form the student's auto_schedule

        Tensor      : list
            the input tensors and the output tensor

        control_f   : the torch function

        shape       : arg for control_f

        time_count  : used for record the running time

        count       : (optional: 10)the number rounds repeat testing

        device_id   : (optional: 0)the id of CPU

        timeout     : (optional: 10.0)time limit for culation
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

    try:
        _, tensors = func(*shape)
        del tensors[-1]
    except Exception as e:
        string = "Failed in torch culation for shape {}\n{}".format(shape, str(e))
        return string
    # Craft input data.
    try:
        input_torch = []

        for tensor in tensors:
            data = np.random.random(
                [int(j) for j in tensor.shape]).astype(np.float32) * 100
            torch_data = torch.tensor(data)
            input_torch.append(torch_data)
    except Exception as e:
        string = "Can't prepare input data for shape {}\n{}".format(shape, str(e))
        return string
    
    torch_args = []
    # TODO use shape length to distinguish conv2d and gemm is foolish
    # No bias if this is convolution
    if len(shape) > 8 and shape[8] == 0:
        torch_args.append(None)
    torch_args.extend(shape[9:])

    try:
        # warm-up
        control_f(*(input_torch + torch_args))

        begin = time.time()
        for i in range(0, count):
            control_f(*(input_torch + torch_args))
        end = time.time()
        torch_time = (end - begin) * 1e3 / count
    except Exception as e:
        string = "Failed in torch culation for shape {}\n{}".format(shape, str(e))
        return string

    return torch_time


def _auto_schedule(auto_schedule_func, func, shape, queue, timeout=20 * 60):
    '''Interface of auto_schedule

        Args:
        -----------------------------
        auto_schedule_func  : auto_schedule function

        func                : conv2d_nchw or gemm

        shape               : args for auto_schedule

        timeout_create      : (optional: 20*60)time limit for auto_schedule_func
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


def _evaluate(student_id, func, shape, time_count, res_path, score_item, unpack_path, target="llvm", dev_id=0, times=10, timeout_create=20 * 60, timeout_cal=10.0):
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

        timeout_cal     : float
            time limit in calculating

        time_count      : Queue
            for testing result transferring
        -----------------------------

        Returns:
        -----------------------------
        '''
    # import student module
    try:
        student_module = load_module('student_module', *find_module(student_id, ['../extract']))
    except ImportError:
        score_list = [0 for i in range(20)]
        write_score(student_id, res_path, score_list, score_item, 'Error in Importing')
        traceback.print_exc()
        print('An error occurs when importing the file as module:', unpack_path)
        return -1

    # scheduling
    s, bufs = _auto_schedule(student_module.auto_schedule, func, shape, time_count, timeout_create)
    if s is None or bufs is None:
        return -1

    # evaluating
    ret = build_and_run(s, bufs, shape, time_count, times, dev_id, target, timeout_cal)
    if ret < 0:
        return -1
    return 0


def evaluate(student_id, func, shape, res_path, score_item, unpack_path, target="llvm", timeout_create=20 * 60, timeout_cal=10.0, times=10, dev_id=0):
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

        times           : (optional: 10) int
            times of calculating in Build_and_Run
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
            student_id, func, shape, time_count, res_path, score_item, unpack_path, target,
            dev_id, times, timeout_create, timeout_cal))
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
    ans = -1
    if not time_count.empty():
        auto_time = time_count.get()
        if isinstance(auto_time, str):
            print("Exceptons occur in shape {}".format(shape))
            print(auto_time)
        else:
            ans = auto_time
    else:
        print("Shape {} can't get results!".format(shape))
    # clean the queue
    time_count.close()
    del time_count
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


def parallel_evaluate(parallel=1):
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

    time_create = 20 * 60
    time_cal = 10.0
    number_test = 10

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
    conv2d_shapes = config.conv_shapes.copy()
    gemm_shapes = config.gemm_shapes.copy()
    np.random.shuffle(conv2d_shapes)
    np.random.shuffle(gemm_shapes)
    score_item = ['gemm_' + str(s) for s in gemm_shapes] + ['conv2d_' + str(s) for s in conv2d_shapes]
    target = 'llvm'

    # for stdout logs
    count_task = 0
    num_tasks = len(total_tasks)
    start_time = time.time()

    # exception info
    # prob_exceptions = ('Import Failure', 'illegal auto_schedule', 'TLE auto_schedule', 'Build Failure', 'TLE run')

    # evaluate func
    def pool_evaluate(time_create, time_cal, number_test, res_path, score_item, unpack_path, shapes, student_id, func, target="llvm"):
        # create process Pool for shapes
        p = NewPool()
        run_time = []
        # exception_stat = [0, 0, 0, 0, 0]
        exception_stat = 0
        sub_procs = []
        for shape in shapes:
            subp = p.apply_async(evaluate, (student_id, func, shape, res_path, score_item, unpack_path, target, time_create, time_cal, number_test))
            sub_procs.append(subp)
        p.close()
        p.join()

        run_time=[]
        for subp in sub_procs:
            case_time = subp.get()
            if case_time == -1:
                exception_stat += 1
            run_time.append(case_time)

        return run_time, exception_stat

    torch_time = []
    for i in gemm_shapes:
        ans = torch_run(batch_gemm, torch_batch_gemm, i, number_test)
        if not isinstance(ans, str):
            torch_time.append(ans)
        else:
            sys.stdout.write(ans+"now exit...")
            sys.stdout.flush()
            return -1

    for i in conv2d_shapes:
        ans = torch_run(conv2d_nchw, torch_conv2d, i, number_test)
        if not isinstance(ans, str):
            torch_time.append(ans)
        else:
            sys.stdout.write(ans+"now exit...")
            sys.stdout.flush()
            return -1

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
        except (ValueError, shutil.ReadError):
            score_list = [0 for i in range(20)]
            write_score(student_id, res_path, score_list, score_item, 'Error in Unpacking')
            sys.stdout.write('An error occurs when unpacking the archive:'+ filezip + '\n')
            sys.stdout.flush()
            continue
            
        # evaluate
        num_gemms = len(gemm_shapes)
        outer = ceil(num_gemms / parallel)
        gemm_ret = []
        gemm_error_count = 0
        for i in range(outer):
            part_gemm_ret, part_gemm_error = pool_evaluate(time_create, time_cal, number_test, res_path, score_item,
                    unpack_path, gemm_shapes[i * parallel:(i+1) * parallel], student_id, batch_gemm, target)
            gemm_ret.extend(part_gemm_ret)
            gemm_error_count += part_gemm_error

        num_convs = len(conv2d_shapes)
        outer = ceil(num_convs / parallel)
        conv_ret = []
        conv_error_count = 0
        for i in range(outer):
            part_conv_ret, part_conv_error = pool_evaluate(time_create, time_cal, number_test, res_path, score_item,
                    unpack_path, conv2d_shapes[i * parallel:(i+1) * parallel], student_id, conv2d_nchw, target)
            conv_ret.extend(part_conv_ret)
            conv_error_count += part_conv_error

        if gemm_error_count + conv_error_count:
            exception_info = ' exception raises in {} cases'.format(gemm_error_count + conv_error_count)
        else:
            exception_info = ' No exceptions'

        tvmtime_list= gemm_ret + conv_ret
        time_list = []
        for i in range(len(tvmtime_list)):
                time_list.append([tvmtime_list[i],torch_time[i]])
        score_list=list(map(score_calculate,time_list))

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

    if time_tvm <= 0:
        return 0.0
    perf_rate = time_torch / time_tvm
    if 0 < perf_rate <= 0.1:
        return 0.0
    elif 0.1 < perf_rate <= 0.2:
        return 0.5
    elif 0.2 < perf_rate <= 0.3:
        return 1.0
    elif 0.3 < perf_rate <= 0.4:
        return 1.5
    elif 0.4 < perf_rate <= 0.5:
        return 2.0
    elif 0.5 < perf_rate <= 0.6:
        return 2.5
    elif 0.7 < perf_rate <= 0.7:
        return 3.0
    elif 0.7 < perf_rate <= 0.8:
        return 3.5
    elif 0.8 < perf_rate <= 0.9:
        return 4.0
    else:
        return 5.0


if __name__ == '__main__':
    parallel_evaluate()
