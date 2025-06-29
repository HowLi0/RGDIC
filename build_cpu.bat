@echo off
echo ================================================
echo RGDIC2 CPU 版本编译脚本
echo ================================================

REM 设置MSVC环境
echo 设置 MSVC 编译环境...
call "D:\Visual Studio\VS  IDE\VC\Auxiliary\Build\vcvars64.bat"
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
del /f /q main_cpu.exe 2>nul
del /f /q *.pdb 2>nul

REM 编译并链接CPU版本（一步完成）
echo 编译 CPU 版本...
cl /MD /O2 /EHsc /IEigen3 /Ifftw3/include /Iopencv/include main_cpu.cpp common_functions.cpp rgdic.cpp icgn_optimizer.cpp neighbor_utils.cpp /link /OUT:main_cpu.exe /LIBPATH:opencv/lib /LIBPATH:fftw3/lib opencv_world4100.lib libfftw3-3.lib libfftw3f-3.lib libfftw3l-3.lib /NODEFAULTLIB:LIBCMT

if errorlevel 1 (
    echo 错误: CPU版本编译失败
    pause
    exit /b 1
)

echo ================================================
echo CPU版本编译成功！生成的可执行文件: main_cpu.exe
echo ================================================

REM 检查文件是否存在
if exist main_cpu.exe (
    echo CPU版本程序已成功编译：main_cpu.exe
    dir main_cpu.exe
) else (
    echo 警告: 未找到 main_cpu.exe 文件
)

echo.
echo 使用说明:
echo 1. 运行 main_cpu.exe 使用合成图像进行测试
echo 2. 运行 main_cpu.exe reference.png deformed.png 处理真实图像
echo 3. 结果将保存在 result/ 文件夹中
echo.

pause
