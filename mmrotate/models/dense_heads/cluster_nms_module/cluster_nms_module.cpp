#include <torch/extension.h>
#include <iostream>

using namespace torch;

void cluster_nms(Tensor& indices, Tensor& keep) {
    Tensor indices_next = indices, indices_prev = indices;
    Tensor b_prev, b_next = at::ones_like(keep);
    // int i = 0;
    do
    {
        // i++;
        b_prev = b_next;
        indices_prev = indices_next;

        b_next = logical_not(any(indices_prev, 0));
        indices_next = __and__(indices, b_next.unsqueeze(1));
    } while (!equal(b_next, b_prev));
    // std::cout << i << std::endl;
    // std::cout << __and__(keep, b_next).nonzero().size(0) << std::endl;
    keep.index({indexing::Slice()}) = __and__(keep, b_next);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_nms", &cluster_nms, "");
}