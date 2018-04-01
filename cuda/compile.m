%compile the cuda and C file

system('nvcc -c pixelAssign.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex pixelAssign.cpp pixelAssign.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c smoothing.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex smoothing.cpp smoothing.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c midsmooth.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex midsmooth.cpp midsmooth.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c Final_smooth.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex Final_smooth.cpp Final_smooth.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c computec.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex computec.cpp computec.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c Cost.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex Cost.cpp Cost.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c colorsm.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex colorsm.cpp colorsm.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c colorsm2.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex colorsm2.cpp colorsm2.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c ModelTransfer.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex ModelTransfer.cpp ModelTransfer.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";
system('nvcc -c FrameTrans.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex FrameTrans.cpp FrameTrans.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c ModelT.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex ModelT.cpp ModelT.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";


system('nvcc -c FindGrad.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex FindGrad.cpp FindGrad.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c ComputeCD.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex ComputeCD.cpp ComputeCD.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";

system('nvcc -c FrameT.cu -ccbin "E:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin"');
mex FrameT.cpp FrameT.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64";



disp('compile finish!');
