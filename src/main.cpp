#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main()
{
    // Example of creating a dummy input tensor.
    torch::Tensor input = torch::rand({1, 3, 224, 224}).to(at::kCUDA); // Adjust the size as per your model input requirements
    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("/home/alejo/Documentos/maestria/embebidos/torch_project/models/resnet18_jit.pt");
        module.to(at::kCUDA); // Move the model to GPU
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model:\n " << e.what() << std::endl;

        return -1;
    }

    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! Moving model to GPU." << std::endl;
        module.to(at::kCUDA); // Move the model to GPU
    }
    else
    {
        std::cout << "Using CPU." << std::endl;
    }

    torch::Tensor output;
    torch::NoGradGuard no_grad; // Ensure that autograd is turned off.
    try
    {
        output = module.forward({input}).toTensor();
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error during the forward pass\n";
        return -1;
    }

    // Example: Print output tensor
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

    return 0;
}
