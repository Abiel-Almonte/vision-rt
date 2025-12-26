#include <torch/torch.h>

torch::Tensor launch_add_relu(
    const torch::Tensor& lhs, // [bs, ch, h, w]
    const torch::Tensor& rhs // [bs, ch, h, w]
);
