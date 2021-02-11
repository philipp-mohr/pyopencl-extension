from pathlib import Path

whole_signed_number_types = ['char', 'short', 'int', 'long']
whole_unsigned_number_types = [f'u{t}' for t in whole_signed_number_types]
whole_number_types = whole_signed_number_types + whole_unsigned_number_types
fractional_number_types = ['half', 'float', 'double']
all_scalar_number_types = whole_number_types + fractional_number_types
vector_type_lengths = [2, 4, 8, 16]
all_vec_number_types = [f'{num}{vec}' for num in all_scalar_number_types
                        for vec in vector_type_lengths]
all_number_types = all_scalar_number_types + all_vec_number_types
auto_gen_dir = Path(__file__).parent


def setup_file_cl_types():
    # preamble_typedefs_np_c = '\n'.join([f'{v} = np.dtype(\'{k}\').type' for k, v in np_to_c_type_name.items()])
    fields = '\n'.join([f'   {t}:np.dtype=cltypes.{t}' for t in all_vec_number_types])
    class_def1 = '@dataclass\n' \
                 'class ClTypesVector:\n' + fields + '\n'

    fields = '\n'.join([f'   {t}:np.dtype=cltypes.{t}' for t in all_scalar_number_types])
    class_def2 = '@dataclass\n' \
                 'class ClTypesScalar:\n' + fields + '\n'

    class_def3 = '@dataclass\n' \
                 'class _ClTypes(ClTypesScalar, ClTypesVector):\n' \
                 '   pass' + '\n'

    class_defs = '\n'.join([class_def1, class_def2, class_def3])

    import_cltypes = 'from pyopencl_extension.modifications_pyopencl import cltypes\n' \
                     'from dataclasses import dataclass\n' \
                     'import numpy as np\n'
    content = '\n'.join([import_cltypes, class_defs])
    with open(auto_gen_dir.joinpath('cl_types.py'), 'w+') as file:
        file.write(content)


def setup_file_include_for_emulation():
    # preamble_typedefs_np_c = '\n'.join([f'{v} = np.dtype(\'{k}\').type' for k, v in np_to_c_type_name.items()])
    python_types = '\n'.join(['int_ = int', 'float_=float'])
    scalar = '\n'.join(
        [f'{t}=TypeHandlerScalar(cltypes.{t})' for t in all_scalar_number_types])
    vec = '\n'.join([f'{t}=TypeHandlerVec(\'{t}\')' for t in all_vec_number_types])
    import_cltypes = 'from pyopencl_extension.modifications_pyopencl import cltypes\n' \
                     'from pyopencl_extension.types.type_handler import *'
    content = '\n'.join([import_cltypes, python_types, scalar, vec])
    with open(auto_gen_dir.joinpath('types_for_emulation.py'), 'w+') as file:
        file.write(content)


if __name__ == '__main__':
    setup_file_include_for_emulation()
    setup_file_cl_types()
