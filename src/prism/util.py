import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def _ArrayModule2SparseArrayModule(array_module: pycd.NDArrayInfo) -> pycd.SparseArrayInfo:
    if array_module == pycd.NDArrayInfo.CUPY:
        sparse_module = pycd.SparseArrayInfo.CUPY_SPARSE
    elif array_module == pycd.NDArrayInfo.DASK:
        sparse_module = pycd.SparseArrayInfo.PYDATA_SPARSE
    else:
        sparse_module = pycd.SparseArrayInfo.SCIPY_SPARSE
    return sparse_module


def _to_sparse_backend(spmat: pyct.SparseArray,
                       sparse_module: pycd.SparseArrayInfo = pycd.SparseArrayInfo.SCIPY_SPARSE):
    if sparse_module == pycd.SparseArrayInfo.CUPY_SPARSE:
        spmat = pycd.SparseArrayInfo.CUPY_SPARSE.module().csr_matrix(spmat)
    elif sparse_module == pycd.SparseArrayInfo.PYDATA_SPARSE:
        spmat = pycd.SparseArrayInfo.PYDATA_SPARSE.module().GCXS(spmat)
    return spmat