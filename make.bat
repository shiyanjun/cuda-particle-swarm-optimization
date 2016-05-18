@echo off
cls
nvcc -m64 -Iinclude -Llib -lglew32 -lfreeglut src/main.cu -o bin/swarm
