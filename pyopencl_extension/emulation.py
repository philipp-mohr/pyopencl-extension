import inspect
import logging
import os
import re
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import indent
from typing import Tuple, List

import numpy as np
from pycparser.c_ast import BinaryOp, ID, IdentifierType, Return, Constant, Assignment, Decl, FuncDef, Node, TypeDecl, \
    FuncCall, ExprList, Compound, ArrayRef, For, If, Cast, ArrayDecl, UnaryOp, While, Break, InitList, Typedef, \
    StructRef, PtrDecl, TernaryOp
from pycparserext.ext_c_parser import FuncDeclExt, PreprocessorLine, OpenCLCParser

__author__ = "piveloper"
__copyright__ = "26.03.2020, piveloper"
__version__ = "1.0"
__email__ = "piveloper@gmail.com"
__doc__ = """This script includes helpful functions to extended PyOpenCl functionality."""

# make sure emulation does not import anything from framework, to avoid circular imports
# framework does depends on emulation and not the other way around.
from pyopencl._cl import LocalMemory

from pyopencl.array import Array
from pytest import mark

from pyopencl_extension.types.funcs_for_emulation import CArray, CArrayVec, init_array
from pyopencl_extension.helpers.general import write_string_to_file

# following lines are used to support functions from <pyopencl-complex.h>
from pyopencl_extension.types.utilities_np_cl import is_vector_type

preamble_buff_t_complex_np = lambda cplx_t: """
{cplx_t}_mul = lambda x, y: {cplx_t}_t(x * y)
{cplx_t}_add = lambda x, y: {cplx_t}_t(x + y)
{cplx_t}_sub = lambda x, y: {cplx_t}_t(x - y)
{cplx_t}_abs = lambda x: {cplx_t}_t(np.abs(x))
{cplx_t}_rmul = lambda x_real, y_cplx: {cplx_t}_t(x_real * y_cplx)
{cplx_t}_new = lambda real, imag: {cplx_t}_t(complex(real, imag))
{cplx_t}_conj = lambda x: np.conj(x)
""".format(cplx_t=cplx_t)

preamble_buff_t_complex64_np = 'cfloat_t = np.dtype(\'complex64\').type\nPI = 3.14159265359\n{}'.format(
    preamble_buff_t_complex_np('cfloat'))
preamble_buff_t_complex128_np = 'cdouble_t = np.dtype(\'complex128\').type\nPI = 3.14159265358979323846\n{}'.format(
    preamble_buff_t_complex_np('cdouble'))


# unparse to python
def py_indent(text):
    return indent(text, 4 * ' ')


# This section contains function that raise error messages if pattern likely causing an erroneous behaviour is detected
def detect_inconsistent_for_loop_header_variable(node: For):
    # case: for(int i=0; t<10;t++){}, raises error if variable in decl, cond and incr do not match
    init_var = node.init.decls[0].name
    if isinstance(node.cond.left, ID):
        cond_var = node.cond.left.name
    else:
        cond_var = node.cond.right.name
    incr_var = node.next.expr.name
    if not init_var == cond_var == incr_var:
        raise ValueError(
            f'Line {node.coord.line}: For loop header contains different variables {init_var=} {cond_var=} {incr_var=}')


# @dataclass
# class Container:
#     unparsed_code: str
#     # https://overiq.com/c-programming-101/increment-and-decrement-operators-in-c/
#     code_unary_operator_prefix: List[str] = None  # e.g. array[++x], y= ++x;  array[--x]; y = --x;
#     code_unary_operator_postfix: List[str] = None  # e.g. array[x++], y= x++;  array[x--]; y = x--;

# https://stackoverflow.com/questions/2673651/inheritance-from-str-or-int
class Container(str):
    def __new__(cls, value, code_unary_operator_prefix: List[str] = None,
                code_unary_operator_postfix: List[str] = None):
        obj = str.__new__(cls, value)
        if code_unary_operator_prefix is None:
            code_unary_operator_prefix = []
        obj.code_unary_operator_prefix = code_unary_operator_prefix
        if code_unary_operator_postfix is None:
            code_unary_operator_postfix = []
        obj.code_unary_operator_postfix = code_unary_operator_postfix
        return obj

    def wrap_code_with_prefix_postfix(self) -> str:
        str_ = self
        if not len(self.code_unary_operator_prefix) == 0:
            str_ = '{}\n{}'.format('\n'.join(self.code_unary_operator_prefix), str_)
        if not len(self.code_unary_operator_postfix) == 0:
            str_ = '{}\n{}'.format(str_, '\n'.join(self.code_unary_operator_postfix))
        return str(str_)

    def only_use_prefix_postfix(self):
        """
        In case of a nested unary operation like array[number++]=something; the result in Python should be
        array[number] = something
        number += 1.

        In case of a separate unary operation like number++; the result in Python should be
        number+=1 but not numbernumber+=1 as it would be we handled like in the nested case.

        Therefore, this function allows to extract only the unary strings.
        :return:
        """
        str_ = ''
        if not len(self.code_unary_operator_prefix) == 0:
            str_ = '{}'.format('\n'.join(self.code_unary_operator_prefix))
        if self.code_unary_operator_postfix is not None:
            str_ = '{}'.format('\n'.join(self.code_unary_operator_postfix))
        return str(str_)

    def add_container_pre_post_fix(self, container: 'Container'):
        self.code_unary_operator_postfix.extend(container.code_unary_operator_postfix)
        self.code_unary_operator_prefix.extend(container.code_unary_operator_prefix)


regex_real_imag_assignment = re.compile(r'([^\[]+)(\[)(.+)(\])(\.)(real|imag)')  # #define PI 3.14159265358979323846\n

names_func_require_work_item = []

names_func_has_barrier = []


class MacroWithArguments:
    """ Explanation:
    Consider following macro with arguments
    #define DFT2_TWIDDLE(a,b,t) { data_t tmp = t(a-b); a += b; b = tmp; }
    which is used e.g. as:
    DFT2_TWIDDLE(u[0],u[8],mul_p0q8);

    1. replace_macro_having_argument_with_function(code_c)
    # replace defines with functions
    # e.g. '# define DFT2_TWIDDLE(a,b,t) { data_t tmp = t(a-b); a += b; b = tmp; }'
    # becomes: 'void DFT2_TWIDDLE(a,b,t){ data_t tmp = t(a-b); a += b; b = tmp; }

    2. c-parser converts function to function to abstract syntax tree model

    3. unparse_macro_node_and_convert_to_string:
        In a first step this macro is converted to python code which yields:
        def DFT2_TWIDDLE(a, b, t, wi: WorkItem = None):
            tmp = data_t(t(a - b))
            a += b
            b = tmp

        To achieve the same behavior as the C macro, we have to execute the python function code in the
        scope where the function DFT2_TWIDDLE is called. The variables in the arguments have to be
        substituted accordingly.
        We get:
        DFT2_TWIDDLE = 'tmp = data_t({t}({a} - {b}))\n{a} += {b}\n{b} = tmp'
        which is used in the emulator like in the following:
        exec(DFT2_TWIDDLE.format(a='u[0]', b='u[8]', t='mul_p0q8'))
    """
    names_py_macro = {}  # dictionary with executable string

    @classmethod
    def replace_with_function(cls, code_c):
        cls.names_py_macro.clear()  # make sure that dict is empty
        if search(regex_func_def_multi_line, code_c):
            for _ in re.findall(regex_func_def_multi_line, code_c):
                def_part_code_c = ''.join(_)
                res = search(regex_func_def_multi_line, def_part_code_c)
                func = f'void {res.group(3)}({res.group(5)}){{{res.group(9)}}}'
                code_c = code_c.replace(def_part_code_c, func)

                cls.names_py_macro[res.group(3)] = 'to be filled with python code'
        return code_c

    @classmethod
    def unparse_macro_node_and_convert_to_string(cls, node):
        macro = _unparse(node.body)
        args = [p.name for p in node.decl.type.args.params]
        for arg in args:
            macro = re.sub(r'(?<![\w])' + arg + r'(?![\w])', f'{{{arg}}}', macro)
        cls.names_py_macro[node.decl.name] = args
        return f'{node.decl.name} = """{macro}"""'


def _unparse_header(args, node):
    if isinstance(node.type, TypeDecl):
        name = _unparse(node.type)
    else:
        raise ValueError('Not implemented')
    return 'def {}({}, wi: WorkItem = None):'.format(name, args)


def _unparse_func_header(node):
    args = ', '.join(['{}'.format(p.name) for p in node.args.params])
    return _unparse_header(args, node)


def _unparse_knl_header(node):
    """
    If kernel header includes local memory argument, the underlying datatype is extracted.
    e.g. __kernel some_knl(__local short *local_mem)
    is converted in emulation .py file as:
    @cl_kernel
    def some_knl(local_mem: short):
    Later, the datatype "short" is extracted to type the array local_mem in emulation mode.
    """
    args = ', '.join(['{}'.format(p.name) if not any(q in ['local', '__local'] for q in p.quals)
                      else f'{p.name}: {_unparse(p.type.type.type.names[0])}'
                      for p in node.args.params])
    return _unparse_header(args, node)


def _unparse(node: Node) -> Container:
    if isinstance(node, FuncDef):
        # check if function is kernel

        if len(node.decl.funcspec) > 0:
            if node.decl.funcspec[0] == '__kernel':
                header = _unparse_knl_header(node.decl.type)
                header = f'@cl_kernel\n{header}'
            else:
                raise ValueError('Func spec not supported')
        else:
            header = _unparse(node.decl.type)

        body = _unparse(node.body)
        final_yield = ''
        if len(node.decl.funcspec) > 0:
            if node.decl.funcspec[0] == '__kernel':
                final_yield = py_indent('yield  # required to model local memory behaviour correctly')
        if final_yield == '':
            function = '{}\n{}'.format(header, py_indent(body))
        else:
            function = '{}\n{}\n{}'.format(header, py_indent(body), final_yield)
        res = function
    elif isinstance(node, FuncDeclExt):
        res = _unparse_func_header(node)
    elif isinstance(node, TypeDecl):
        res = node.declname
    elif isinstance(node, FuncCall):
        if isinstance(node.name, ID):
            func_name = node.name.name
            # todo: use funcs_for_cl_emulation.py for adding support for all cl functions
            if func_name.startswith('convert_'):
                """
                alternative:
                def convert(value, dtype='int32'):
                    return np.dtype(dtype).type(value)
                """
                type_name = re.search(r'(convert_)([\w]+)', node.name.name).group(2)
                res = '{}({})'.format(type_name, _unparse(node.args))
            elif 'cos' == func_name:
                res = 'np.cos({})'.format(_unparse(node.args))
            elif 'sin' == func_name:
                res = 'np.sin({})'.format(_unparse(node.args))
            elif func_name in translation_cl_work_item_functions:
                res = 'wi.{}[{}]'.format(translation_cl_work_item_functions[func_name], _unparse(node.args))
            elif 'barrier' == func_name:
                if isinstance((_ := node.args.exprs[0]), BinaryOp):
                    barrier_name = f'{node.args.exprs[0].left.name}|{node.args.exprs[0].right.name}'
                else:
                    barrier_name = node.args.exprs[0].name
                res = f'\nwi.scope = locals()  # saves reference to objects in scope for debugging other wi in wg \n' \
                      f'yield  # yield models the behaviour of barrier({barrier_name})\n'
            else:
                if func_name in (_ := MacroWithArguments.names_py_macro):
                    args = _unparse(node.args).split(', ')
                    format_arg = ', '.join([f'{macro_arg}="{args[i]}"' for i, macro_arg in enumerate(_[func_name])])
                    res = f'exec({func_name}.format({format_arg}))'
                elif func_name in names_func_require_work_item:
                    res = '{}({}, wi=wi)'.format(_unparse(node.name), _unparse(node.args))
                    if func_name in names_func_has_barrier:
                        res = f'(yield from {res})'
                else:
                    res = '{}({})'.format(_unparse(node.name), _unparse(node.args))
        else:
            res = '{}({})'.format(_unparse(node.name), _unparse(node.args))
    elif isinstance(node, ExprList):
        res = ', '.join([_unparse(expr) for expr in node.exprs])
    elif isinstance(node, Compound):  # e.g. body of a function or a for loop
        # compound = [_unparse(block_item) for block_item in node.block_items]
        # loop makes debugging easier. Set breakpoint in loop, then remove BP and step over to get to target line
        compound = []
        for block_item in node.block_items:
            if isinstance(block_item, UnaryOp):  # number++; should translate to number +=1 and not numbernumber+=1
                compound.append(_unparse(block_item).only_use_prefix_postfix())
            else:
                compound.append(_unparse(block_item).wrap_code_with_prefix_postfix())
        res = '\n'.join(compound)
    elif isinstance(node, ArrayRef):
        # if isinstance(node.subscript, UnaryOp):  # e.g. array[pos++] = 5; -> pos+=1; \n array[pos] = 5;
        #     unary_op = _unparse(node.subscript)
        #     res = '{}[{}]'.format(_unparse(node.name), node.subscript.expr.name)
        # else:
        array_index = _unparse(node.subscript)
        res = Container('{}[{}]'.format(_unparse(node.name), array_index))
        res.add_container_pre_post_fix(array_index)
    elif isinstance(node, For):
        # only for loop considered of style: for(int i=start; i<=stop; i++)
        #                                 or for(int i>=stop; i>=0; i--)
        detect_inconsistent_for_loop_header_variable(node)

        var = node.init.decls[0].name
        right = _unparse(node.cond.right)
        left = _unparse(node.cond.left)
        cond_op = node.cond.op
        if right == var:  # e.g. 5>=i -> i<= 5
            right = left
            left = var
            cond_op = {'!=': '!=',
                       '<=': '>=',
                       '>=': '<=',
                       '>': '<',
                       '<': '>'}[cond_op]
        stop = right
        op = node.next.op
        stop = {'p++': {'!=': stop,
                        '<': stop,
                        '<=': stop + '+1'},
                'p--': {'!=': stop,
                        '>': stop,
                        '>=': stop + '-1'}}[op][cond_op]
        if op == 'p++':
            step = 1
        elif op == 'p--':
            step = -1
        else:
            raise NotImplementedError()
        loop_header = '{var} in range({start}, {stop}, {step})'.format(var=var,
                                                                       start=_unparse(node.init.decls[0].init),
                                                                       stop=stop,
                                                                       step=step)
        loop_body = _unparse(node.stmt)
        res = 'for {}:\n{}'.format(loop_header, py_indent(loop_body))
    elif isinstance(node, While):
        res = 'while {}:\n{}'.format(_unparse(node.cond), py_indent(_unparse(node.stmt)))
    elif isinstance(node, Break):
        res = 'break'
    elif isinstance(node, If):
        if_true_cond = '{}'.format(_unparse(node.cond))
        if_true_body = _unparse(node.iftrue)
        if_true = 'if {}:\n{}'.format(if_true_cond, py_indent(if_true_body))
        if node.iffalse is not None:
            else_body = _unparse(node.iffalse)
            else_ = 'else:\n{}'.format(py_indent(else_body))
            res = '{}\n{}'.format(if_true, else_)
        else:
            res = if_true
    elif isinstance(node, Decl):
        if len(node.quals) > 0 and node.quals[0] == 'private':  # private int indices_states_prior[ALPHABET_SIZE];
            res = _unparse(node.type)
        elif isinstance(node.type, ArrayDecl):
            if node.init is None:  # real_t p_x[2];
                res = _unparse(node.type)
            elif isinstance(node.init, InitList):  # real_t p_x[2]={0.0};
                # todo: if array /var is not inialized assign random values or give warning
                array_decl = _unparse(node.type)
                array_fill = [_unparse(item) for item in node.init.exprs]
                # todo: deal with a[5]={1}->a=[1,0,0,0,0} node.type.dim.value == 1 and len(array_fill) == 1 and int(array_fill) != 0:
                if len(array_fill) == 1 and array_fill[0] in ['0', '0.0']:
                    res = '{}\n{}.fill({})'.format(array_decl, node.type.type.declname, array_fill[0])
                else:
                    raise ValueError('Currently array initializer with multiple or non zero values not supported')
        elif node.init is None:
            if isinstance(node.type, PtrDecl):  # e.g. float *x;
                res = f'{node.name} = {node.type.type.type.names[0]}(0)'
            else:  # e.g. float x;
                res = f'{node.name} = {node.type.type.names[0]}(0)'
        else:  # int gid1=get_global_id(0);
            if isinstance(node.type, PtrDecl):
                type_cl = node.type.type.type.names[0]
            else:
                type_cl = node.type.type.names[0]
            res = '{} = {}({})'.format(node.name, type_cl, _unparse(node.init))
    elif isinstance(node, ArrayDecl):
        if len(node.type.quals) > 0 and node.type.quals[0] in ['local', '__local']:
            res = "{name} = local_memory(wi, '{name}', lambda: init_array({dim}, {dtype}))".format(
                name=_unparse(node.type),
                dim=_unparse(node.dim),
                dtype=node.type.type.names[0])
        else:
            res = f'{_unparse(node.type)} = init_array({_unparse(node.dim)}, {node.type.type.names[0]})'
    elif isinstance(node, Assignment):
        left = _unparse(node.lvalue)
        right = _unparse(node.rvalue)
        if search(regex_real_imag_assignment, left):
            groups = search(regex_real_imag_assignment, left).groups()
            res = Container(f'set_{groups[5]}(ary={groups[0]}, idx={groups[2]}, value={right})')
        else:
            res = Container('{} {} {}'.format(left, _unparse(node.op), right))
            res.add_container_pre_post_fix(left)
            res.add_container_pre_post_fix(right)
    elif isinstance(node, Constant):
        res = node.value
    elif isinstance(node, Return):
        res = 'return ' + _unparse(node.expr)
    elif isinstance(node, BinaryOp):
        left = _unparse(node.left)
        right = _unparse(node.right)
        # todo: see test_pointer_arithmetics
        # if isinstance(node.left, ID) and node.op in ['+']:
        #     res = f'{node.left.name}[{node.op}:]'
        # elif isinstance(node.right, ID) and node.op in ['+']:
        #     res = f'{node.right.name}[{node.op}:]'
        if node.op in ['%']:
            res = Container(f'c_modulo({left},{right})')
        else:
            if node.op in ['&&']:  # && not supported in python
                node_op = '&'
            elif node.op in ['||']:
                node_op = 'or'
            else:
                node_op = node.op
            # following lines reduce unnecessary brackets like 1+2 is not translated to (1+2) in Python equivalent
            if isinstance(node.right, BinaryOp):
                right = Container(f'({right})', **right.__dict__)
            if isinstance(node.left, BinaryOp):
                left = Container(f'({left})', **left.__dict__)
            res = Container('{} {} {}'.format(left, node_op, right))
        res.add_container_pre_post_fix(left)
        res.add_container_pre_post_fix(right)
    elif isinstance(node, ID):
        res = node.name
    elif isinstance(node, IdentifierType):
        res = ''
    elif isinstance(node, str):
        res = node
    elif isinstance(node, Cast):
        type_cl = node.to_type.type.type.names[0]
        res = '{}({})'.format(type_cl, _unparse(node.expr))
    elif isinstance(node, UnaryOp):
        if node.op == 'p++':
            res = Container(node.expr.name, code_unary_operator_postfix=['{} += 1'.format(_unparse(node.expr))])
            # res = '{} += 1'.format(_unparse(node.expr))
        elif node.op == 'p--':
            res = Container(node.expr.name, code_unary_operator_postfix=['{} -= 1'.format(_unparse(node.expr))])
        elif node.op == '++':
            res = Container(node.expr.name, code_unary_operator_prefix=['{} += 1'.format(_unparse(node.expr))])
        elif node.op == '--':
            res = Container(node.expr.name, code_unary_operator_prefix=['{} -= 1'.format(_unparse(node.expr))])
        elif node.op == '!':
            res = f'not {_unparse(node.expr)}'
        else:
            res = '{}{}'.format(node.op, _unparse(node.expr))
    elif isinstance(node, StructRef):
        res = f'{_unparse(node.name)}.{_unparse(node.field)}'
    elif isinstance(node, TernaryOp):
        res = f'{_unparse(node.iftrue)} if {_unparse(node.cond)} else {_unparse(node.iffalse)}'
    elif node is None:
        res = ''
    else:
        raise ValueError('Node type not considered')
    if not isinstance(res, Container):
        res = Container(res)
    return res


def unparse_function_node(node: Node) -> str:
    return str(_unparse(node))


# put work item functions in tuples such that they are immediately visible when hovering with cursor in debug mode
translation_cl_work_item_functions = {
    'get_global_id': 'global_id',
    'get_local_id': 'local_id',
    'get_global_size': 'global_size',
    'get_local_size': 'local_size',
    'get_num_groups': 'num_groups',
    'get_group_id': 'group_id',
    'get_work_dim': 'work_dim',
}


@dataclass
class WorkItem:
    global_id: Tuple[int]
    global_size: Tuple[int]
    local_size: Tuple[int]
    local_memory_collection: List[dict]
    _local: dict = field(init=False, default=None)
    _group_id_lin: int = None  #
    _scope: dict = None
    work_items: List['WorkItem'] = None
    local_id: Tuple[int] = None
    num_groups: Tuple[int] = None
    group_id: Tuple[int] = None
    work_dim: Tuple[int] = None

    def __post_init__(self):
        n_dim = self.get_work_dim()
        self.work_dim = n_dim
        self.local_id = tuple([self.get_local_id(i) for i in range(n_dim)])
        self.num_groups = tuple([self.get_num_groups(i) for i in range(n_dim)])
        self.group_id = tuple([self.get_group_id(i) for i in range(n_dim)])
        self._group_id_lin = compute_linear_idx(self.group_id, self.num_groups)

    @property
    def info(self):
        n_dim = len(self.global_size)
        local_id = tuple(self.get_local_id(dim) for dim in range(n_dim))
        return f'global_id={self.global_id} local_id={local_id}'

    def get_vars_in_wg(self, name: str):
        """
        Use case: Consider x is a private scoped array of a work item. Then np.vstack(wi.get_scope_vars_in_wg('x'))
        will build a 2d array where all privates scoped arrays named 'x' of the work items of the work group are
        stacked. This feature facilitates debugging where computations are done from cooperating work items in a work
        group.

        :param name:
        :return:
        """
        return np.vstack([_wi.scope[name] for _wi in self.wi_in_wg])

    @property
    def scope(self):
        # remove circular reference to self
        return {k: v for k, v in self._scope.items() if k != 'wi' or k.startswith('__')}

    @scope.setter
    def scope(self, value):
        self._scope = value

    @property
    def wi_in_wg(self):
        return [wi for wi in self.work_items if wi._group_id_as_tuple == self._group_id_as_tuple]

    @property
    def _group_id_as_tuple(self):
        n_dims = len(self.global_size)
        return tuple(self.get_group_id(dim) for dim in range(n_dims))

    @property
    def local(self) -> dict:
        if self._local is None:
            # Use linear index for work group, to retrieve local memory of particular workgroup.
            # All local memory is modeled by self.local_memory_collection
            self._local = self.local_memory_collection[self._group_id_lin]
        return self._local

    def get_global_id(self, dim):
        return self.global_id[dim]

    def get_global_size(self, dim):
        return self.global_size[dim]

    def get_local_id(self, dim) -> int:
        return self.global_id[dim] % self.local_size[dim]

    def get_local_size(self, dim):
        return self.local_size[dim]

    def get_num_groups(self, dim):
        return int(self.global_size[dim] / self.local_size[dim])

    def get_group_id(self, dim):
        return int(self.global_id[dim] / self.local_size[dim])  # return self.local_size[dim]

    def get_global_offset(self, dim):
        raise NotImplementedError()

    def get_work_dim(self):
        return len(self.global_size)


def local_memory(wi: WorkItem, name: str, lambda_array):
    if name not in wi.local:
        wi.local[name] = lambda_array()
    return wi.local[name]


def cl_kernel(kernel):
    """
    Decorater for kernel function emulated with Python.

    :param kernel:
    :return:
    """

    def wrapper_loop_over_grid(global_size, local_size=None, *args):
        args_python = [arg.get().ravel() if isinstance(arg, Array) else arg for arg in args]
        num_work_groups = 1
        if local_size is None:
            local_size = global_size
        for i in range(len(global_size)):
            num_work_groups *= int(global_size[i] / local_size[i])
        local_memory_collection = [{} for _ in range(num_work_groups)]

        work_items = [WorkItem(global_id=gid, global_size=global_size, local_size=local_size,
                               local_memory_collection=local_memory_collection)
                      for gid in np.ndindex(global_size)]

        # filter for local memory arguments and initialize local memory for each work group
        _args_python = []
        LocalArg = namedtuple('LocalArg', 'name')
        for position, arg in enumerate(args_python):
            if isinstance(arg, LocalMemory):
                arg_name = (_ := inspect.getfullargspec(kernel)).args[position]
                arg_type = _.annotations[arg_name]
                for loc_mem in local_memory_collection:
                    loc_mem[arg_name] = init_array(size=int(arg.size / np.dtype(arg_type.dtype).itemsize),
                                                   type_c=arg_type)
                _args_python.append(LocalArg(arg_name))
            else:
                _args_python.append(arg)
        args_python = _args_python

        def decide_ary_view(arg):
            if is_vector_type(arg.dtype):
                return arg.view(CArrayVec)
            else:
                return arg.view(CArray)

        args_python = [decide_ary_view(arg) if isinstance(arg, np.ndarray) else arg
                       for arg in args_python]
        for wi in work_items:
            wi.work_items = work_items

        # Assign work items to individual work groups. Items of a work group can by synchronized using barriers.
        # However, synchronization between work groups is not possible according to OpenCl standard.
        # https://stackoverflow.com/questions/6890302/barriers-in-opencl
        work_items_per_wg = [[] for _ in range(num_work_groups)]
        for wi in work_items:
            work_items_per_wg[compute_linear_idx(wi.group_id, wi.num_groups)].append(wi)

        def prepare_args(work_item):
            return [work_item.local[arg.name] if isinstance(arg, LocalArg) else arg for arg in args_python]

        for i_wg in range(num_work_groups):
            blocking_kernels = [kernel(*prepare_args(work_item), wi=work_item)
                                for work_item in work_items_per_wg[i_wg]]
            while True:
                try:
                    [next(knl) for knl in blocking_kernels]
                except StopIteration:
                    break
        [arg.set(args_python[idx]) if isinstance(arg, Array) else arg for idx, arg in enumerate(args)]

    return wrapper_loop_over_grid


# test regex at https://regexr.com/
regex_constant = re.compile(r'#define[ ]+([\w]+)[ ]+([\w\.\-e]+)')  # #define PI 3.14159265358979323846\n
regex_func_def = re.compile(r'#define[ ]+([\w]+)\(([\w, ]+)\)[ ]+([\w\.\(\)*\-+/ ]+)')  # #define MUL(x,y) (x*y)\n

# #define DFT2_TWIDDLE(a,b,t) { data_t tmp = t(a-b); a += b; b = tmp; }
regex_func_def_multi_line = re.compile(r'(#define)([ ]+)([\w]+)(\()([\w, ]+)(\))([ ]+)(\{)(.+)(\})')
regex_include = re.compile(r'#include[ ]+<([\w\-\.]+>)')  # '#include <pyopencl-complex.h>\n
regex_numbers_with_conversion_characater = re.compile(r'([\d.]+)([fd])')

search = lambda regex, text: re.search(regex, text)


def unparse_preprocessor_line(line: PreprocessorLine) -> str:
    """
    contents = '#define PI 3.14159265358979323846\n'
    contents = '#define MUL(x,y) (x*y)\n'
    contents = '#define NUM 1e-6\n'

    :param preprocessor_line:
    :return:
    """
    contents = line.contents

    if res := search(regex_numbers_with_conversion_characater, contents):
        replacement = res.group(1)
        contents = re.sub(regex_numbers_with_conversion_characater, replacement, contents)
    if res := search(regex_constant, contents):
        search_convert_t = re.search(r'(convert_)([\w]+)\((.*)\)', contents)
        if search_convert_t:  # catch case e.g. LLR_MAX convert_float(200.0)
            type_name = search_convert_t.group(2)
            value = f'{type_name}({search_convert_t.group(3)})'
        else:
            value = res.group(2)
        name = res.group(1)
        return '{} = {}'.format(name, value)  # 'PI = 3.14159265358979323846'
    elif res := search(regex_func_def, contents):
        name = res.group(1)
        args = res.group(2)
        body = res.group(3)

        search_convert_t = re.search(r'(convert_)([\w]+)', name)
        # todo: this type of definition is not used since unparse function replaces convert_t already
        # alternative: delete define statement instead
        if search_convert_t:
            type_name = search_convert_t.group(2)
            body = '{}({})'.format(type_name, args)
        return '{} = lambda {}: {}'.format(name, args, body)  # 'MUL = lambda x, y: x * y'
    elif search(regex_include, contents):
        # until now no case occured where python needs to include something like opencl
        return ''
    else:
        raise ValueError('Preprocessing line content not covered with Regex')


def unparse_type_def_node(node: Typedef):
    type_name_c = node.type.type.names[0]
    # return '{} = np.dtype(ClTypes.{}).type'.format(node.name, type_name_c)
    return f'{node.name} = {type_name_c}'


def search_for_barrier(code_c, ast):
    """
    Searches for barriers inside of functions.
    e.g. consider following c code (Pseudocode):
    int func_nested(){
        barrier(...);

    int func_parent(){
        func_nested()

    __kernel kernel_func(){
        func_parent

    After conversion, the barrier call is emulated by making use of yield and yield from.

    def func_nested():
        barrier(...);

    int func_parent():
        return (yield from func_nested())

    __kernel kernel_func():
        (yield from func_parent)
    """
    code_c = code_c.split('\n')
    lines_with_barrier = [line for line, line_content in enumerate(code_c)
                          if 'barrier(CLK_' in line_content]

    def get_start_line(_):
        if isinstance(_, FuncDef):
            return _.coord.line
        elif isinstance(_, list):
            return _[0].coord.line

    line_endings = [get_start_line(_) for _ in ast.ext][1:] + [len(code_c)]
    funcs = [{'name': _.decl.name,
              'start_line': _.coord.line,
              'end_line': line_endings[i],
              'is_kernel': any(['kernel' in spec for spec in _.decl.funcspec])}
             for i, _ in enumerate(ast.ext) if isinstance(_, FuncDef)]
    # or funcs with nested func that has barrier
    funcs_with_barrier = [_ for _ in funcs if
                          any(_['start_line'] < line < _['end_line'] for line in lines_with_barrier)]
    funcs_nested_barrier = [_ for _ in funcs if any(re.search(r'(\W)(' + func['name'] + r')(\W)',
                                                              ''.join(code_c[_['start_line']:_['end_line']-1]))
                                                    for func in funcs_with_barrier)]

    return [_['name'] for _ in funcs_with_barrier + funcs_nested_barrier if not _['is_kernel']]


def unparse_c_code_to_python(code_c: str) -> str:
    # todo prevents files: https://stackoverflow.com/questions/12644902/how-to-prevent-table-regeneration-in-ply
    # yacc.yacc(debug=False, write_tables=False)
    code_c = re.sub('#define[ ]+TP_ROOT[ ]+(cfloat|cdouble])[ ]*(\n)', '', code_c)  # removes TP_ROOT = cfloat

    p = OpenCLCParser(lex_optimize=False, yacc_optimize=False)
    os.remove('yacctab.py')
    # remove block comments like /* some comment */ since other p.parse throws parsing error
    code_c = re.sub(r'\/\*(\*(?!\/)|[^*])*\*\/', '', code_c)
    code_c = code_c.replace('#pragma unroll', '')

    code_c = MacroWithArguments.replace_with_function(code_c)

    from pyopencl_extension.framework import preamble_activate_complex_numbers
    code_c = code_c.replace(preamble_activate_complex_numbers, '')
    from pyopencl_extension.framework import preamble_activate_double
    code_c = code_c.replace(preamble_activate_double, '')
    code_c = code_c.replace('__const', '')  # todo: create constant array class which raises error when writing to
    # todo: comments can be extracted using line numbers. Nodes in abstract syntax tree provide coords for reinsertion
    ast = p.parse(code_c)  # abstract syntax tree, why no comments? --> https://github.com/eliben/pycparser/issues/124
    module_py = []
    header = """
from typing import Tuple
from pyopencl_extension.emulation import cl_kernel, WorkItem, local_memory
from pyopencl_extension.types.funcs_for_emulation import *
from pyopencl_extension.types.utilities_np_cl import Types, c_to_np_type_name_catch
import numpy as np
            """
    module_py.append(header)
    if 'cfloat' in code_c:
        module_py.append(preamble_buff_t_complex64_np)
    elif 'cdouble' in code_c:
        module_py.append(preamble_buff_t_complex128_np)
    # module_py.append(preamble_cl_funcs_to_lambdas)

    # find funcs that contain barrier(CLK_LOCAL_MEM_FENCE) and therefore require yield from
    names_func_has_barrier.clear()
    names_func_has_barrier.extend(search_for_barrier(code_c, ast))

    names_func_require_work_item.clear()
    names_func_require_work_item.extend([node.decl.name for node in ast.ext if isinstance(node, FuncDef)])
    for node in ast.ext:
        if type(node) == list:
            if len(node) == 1:
                if type(node[0]) == PreprocessorLine:
                    module_py.append(unparse_preprocessor_line(node[0]))
        if type(node) == Typedef:
            module_py.append(unparse_type_def_node(node))
        if isinstance(node, FuncDef):
            module_py.append('\n')
            if node.decl.name in MacroWithArguments.names_py_macro:  # for explanation see comment below names_macro_func_def
                module_py.append(MacroWithArguments.unparse_macro_node_and_convert_to_string(node))
            else:
                module_py.append(unparse_function_node(node))

    code_py = '\n'.join(module_py)
    code_py = code_py + '\n'
    # todo: deal with complex header
    # if 'cfloat_t' in code_c:
    #     preamble_buff_t = preamble_buff_t_complex64_np
    # elif 'cdouble_t' in code_c:
    #     preamble_buff_t = preamble_buff_t_complex128_np
    # else:
    #     preamble_buff_t = preamble_buff_t_real_np
    #
    # preamble_buff_t = '{}\n\n{}'.format(preamble_buff_t, preamble_cl_funcs_to_lambdas)
    return code_py


b_use_existing_file_for_emulation = False


def set_b_use_existing_file_for_emulation(value: bool):
    # hack solution for making changes in generated python cl program directly. When rerunning, those changes are not
    # overridden
    global b_use_existing_file_for_emulation
    b_use_existing_file_for_emulation = value


def create_py_file_and_load_module(code_py: str, file: str = None):
    """
    Creates .py file from code_py and loads created module such that debugger can jump into code.
    :param code_py: Python code snippet in string format
    :param file: Either full path or path relative to working directory without including file type name
    :return:
    """
    if file is None:
        file = str('cl_py_modules/debug_file')
    path_code_py = '{}.py'.format(file)
    # hack solution for making changes in generated python cl program directly. When rerunning, those changes are not
    # overridden
    if Path(path_code_py).exists() and b_use_existing_file_for_emulation:
        pass
    else:
        write_string_to_file(code_py, path_code_py)

    import importlib.util
    spec = importlib.util.spec_from_file_location(file, str(path_code_py))
    program_python = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program_python)
    return program_python


def compute_linear_idx(idx_tuple, dimensions)->int:
    idx = 0
    n_dim = len(dimensions)
    for i in range(n_dim):
        offset = 1
        for j in range(i + 1, n_dim):
            offset *= int(dimensions[j])  # self.get_global_size(j) / self.get_local_size(j)
        idx += idx_tuple[i] * offset
    return idx


def compute_tuple_idx(idx_lin, dimensions):
    n_dim = len(dimensions)
    idx_tuple = [0] * n_dim
    for _i in range(n_dim):
        i = n_dim - 1 - _i
        idx_lin, idx_tuple[i] = divmod(idx_lin, dimensions[i])
    return tuple(idx_tuple)


@mark.parametrize('idx_tuple,dimensions, idx_lin_ref', [((1, 2, 0), (2, 4, 2), 12),
                                                        ((1, 2), (2, 4), 6),
                                                        ((1,), (2,), 1)])
def test_compute_tuple_from_idx_linear(idx_tuple, dimensions, idx_lin_ref):
    idx_lin = compute_linear_idx(idx_tuple, dimensions)
    assert idx_lin == idx_lin_ref
    idx_tuple2 = compute_tuple_idx(idx_lin, dimensions)
    assert idx_tuple == idx_tuple2
