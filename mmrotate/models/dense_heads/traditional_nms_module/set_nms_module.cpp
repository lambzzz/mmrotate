#include <torch/extension.h>

using namespace torch;

void keepbbox(Tensor& indices, Tensor& keep, int len) {
    for (int i = 0; i < len; i++)
    {
        if (keep[i].item<bool>())
        {
            keep.index({indexing::Slice(i+1, indexing::None)}) = __and__(keep.index({indexing::Slice(i+1, indexing::None)}), indices.index({i, indexing::Slice(i+1, indexing::None)}));
        }
        
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("keepbbox", &keepbbox, "");
}