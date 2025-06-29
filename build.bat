@echo off
echo ================================================
echo RGDIC2 CUDA 项目编译脚本
echo ================================================

REM 设置MSVC环境
echo 设置 MSVC 编译环境...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul
    if errorlevel 1 (
        call "D:\Visual Studio\VS  IDE\VC\Auxiliary\Build\vcvars64.bat" 2>nul
        if errorlevel 1 (
            echo 错误: 无法找到 Visual Studio 安装路径
            echo 请修改此脚本中的 Visual Studio 路径
            pause
            exit /b 1
        )
    )
)

echo MSVC 环境已设置

REM 清理之前的编译文件
echo 清理之前的编译文件...
del /f /q *.obj 2>nul
del /f /q *.exe 2>nul
del /f /q *.pdb 2>nul

REM 编译 CUDA kernels
echo 编译 CUDA kernels...
nvcc -c cuda_dic_kernels.cu -I Eigen3 -I opencv/include -I fftw3/include --compiler-options "/MD /O2" -arch=sm_75 -o cuda_dic_kernels.obj
if errorlevel 1 (
    echo 错误: CUDA kernels 编译失败
    pause
    exit /b 1
)

REM 编译 CUDA C++ 源文件
echo 编译 CUDA C++ 源文件...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" cuda_rgdic.cpp cuda_dic_kernel.cpp
if errorlevel 1 (
    echo 错误: CUDA C++ 源文件编译失败
    pause
    exit /b 1
)

REM 编译 CPU 源文件
echo 编译 CPU 源文件...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" rgdic.cpp icgn_optimizer.cpp neighbor_utils.cpp
if errorlevel 1 (
    echo 错误: CPU 源文件编译失败
    pause
    exit /b 1
)

REM 编译主程序
echo 编译主程序...
cl /c /MD /O2 /EHsc /DUSE_CUDA /IEigen3 /Ifftw3/include /Iopencv/include /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" main.cpp
if errorlevel 1 (
    echo 错误: 主程序编译失败
    pause
    exit /b 1
)

REM 链接程序
echo 链接程序...
link /OUT:main_cuda.exe main.obj rgdic.obj icgn_optimizer.obj neighbor_utils.obj cuda_rgdic.obj cuda_dic_kernel.obj cuda_dic_kernels.obj /LIBPATH:opencv/lib /LIBPATH:fftw3/lib /LIBPATH:"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" opencv_world4100.lib libfftw3-3.lib libfftw3f-3.lib libfftw3l-3.lib cudart.lib cublas.lib curand.lib cusolver.lib cusparse.lib /NODEFAULTLIB:LIBCMT
if errorlevel 1 (
    echo 错误: 程序链接失败
    pause
    exit /b 1
)

echo ================================================
echo 编译成功！生成的可执行文件: main_cuda.exe
echo ================================================

REM 检查文件是否存在
if exist main_cuda.exe (
    echo 主程序已成功编译：main_cuda.exe
    dir main_cuda.exe
) else (
    echo 警告: 未找到 main_cuda.exe 文件
)

echo.
echo 使用说明:
echo 1. 运行 main_cuda.exe 使用合成图像进行测试
echo 2. 运行 main_cuda.exe reference.png deformed.png 处理真实图像
echo 3. 结果将保存在 result/ 文件夹中
echo.

pause
