rem call VsDevCmd to setup environment
call %1 -arch=x64
rem call nvcc
call %2 %3 -o %4