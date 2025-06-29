@echo off
echo Setting up MSVC environment...
call "D:\Visual Studio\VS  IDE\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Compiling high-precision CUDA kernels...
nvcc -c cuda_dic_kernels_precision.cu -I Eigen3 -I opencv/include -I fftw3/include --compiler-options "/MD /O2" -arch=sm_75 -o cuda_dic_kernels_precision.obj
if %errorlevel% neq 0 goto error

echo.
echo Compiling CUDA C++ sources...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" cuda_rgdic.cpp cuda_dic_kernel_precision.cpp
if %errorlevel% neq 0 goto error

echo.
echo Compiling CPU sources...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" rgdic.cpp icgn_optimizer.cpp neighbor_utils.cpp
if %errorlevel% neq 0 goto error

echo.
echo Compiling common functions...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" common_functions.cpp
if %errorlevel% neq 0 goto error

echo.
echo Compiling CUDA main program...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" main_cuda.cpp
if %errorlevel% neq 0 goto error

echo.
echo Linking CUDA program...
link /OUT:main_cuda.exe main_cuda.obj common_functions.obj rgdic.obj icgn_optimizer.obj neighbor_utils.obj cuda_rgdic.obj cuda_dic_kernel_precision.obj cuda_dic_kernels_precision.obj /LIBPATH:opencv/lib /LIBPATH:fftw3/lib /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" opencv_world4100.lib libfftw3-3.lib libfftw3f-3.lib libfftw3l-3.lib cudart.lib cublas.lib curand.lib cusolver.lib cusparse.lib /NODEFAULTLIB:LIBCMT
if %errorlevel% neq 0 goto error

echo.
echo High-precision CUDA build successful! main_cuda.exe created.
goto end

:error
echo.
echo Build failed!
exit /b 1

:end
