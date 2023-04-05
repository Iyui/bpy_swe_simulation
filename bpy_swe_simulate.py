import bpy
import numpy as np
import math
import bmesh

# 模拟参数和之前的计算代码保持不变
# 模拟参数
g = 9.81
dt = 0.1
t_steps = 100
grid_size = 100
nu = 1e-6
# 定义高斯函数
def gaussian(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

# 生成网格点
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

# 设置初始水面高度
x0, y0 = grid_size // 2, grid_size // 2
sigma = 10
h = gaussian(x, y, x0, y0, sigma)*100

# 初始化速度
u = np.zeros((grid_size, grid_size, 2))
H = np.zeros((grid_size, grid_size))
a_ext = np.zeros((grid_size, grid_size, 2))

# 计算速度场的Laplacian
def laplacian(u):
    laplacian_u = np.zeros_like(u)
    for dim in range(2):
        laplacian_u[:, :, dim] = (np.roll(u[:, :, dim], -1, axis=1 - dim) - 2 * u[:, :, dim] + np.roll(u[:, :, dim], 1, axis=1 - dim))
    return laplacian_u

# 进行数值求解
for t in range(t_steps):
    div_u = np.gradient(u[:, :, 0])[1] + np.gradient(u[:, :, 1])[0]
    h -= dt * div_u

    grad_h = np.array([np.gradient(h)[1], np.gradient(h)[0]])
    grad_H = np.array([np.gradient(H)[1], np.gradient(H)[0]])

    lap_u = laplacian(u)

    for step in range(2):
        u[:, :, step] -= dt * (g * (grad_h[step] + grad_H[step]) + a_ext[:, :, step]) + dt * nu * lap_u[:, :, step]
# 在 Blender 中创建模拟结果的可视化

# 删除所有网格对象
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# 创建一个平面对象
bpy.ops.mesh.primitive_plane_add(size=grid_size - 1, enter_editmode=False, align='WORLD', location=((grid_size - 1) / 2, (grid_size - 1) / 2, 0))
plane = bpy.context.active_object
plane.name = "Simulation_Plane"

# 细分平面
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.subdivide(number_cuts=grid_size - 2)
bpy.ops.object.mode_set(mode='OBJECT')

# 创建形状关键帧
plane.shape_key_add(name="Basis")

# 在每个时间步长上执行模拟并创建对应的形状关键帧
for t in range(t_steps):
    div_u = np.gradient(u[:, :, 0])[1] + np.gradient(u[:, :, 1])[0]
    h -= dt * div_u

    grad_h = np.array([np.gradient(h)[1], np.gradient(h)[0]])
    grad_H = np.array([np.gradient(H)[1], np.gradient(H)[0]])

    lap_u = laplacian(u)

    for step in range(2):
        u[:, :, step] -= dt * (g * (grad_h[step] + grad_H[step]) + a_ext[:, :, step]) + dt * nu * lap_u[:, :, step]

    # 创建模拟结果的形状关键帧
    simulation_key = plane.shape_key_add(name=f"Simulation_{t}")
    vertices = plane.data.shape_keys.key_blocks[f"Simulation_{t}"].data

    # 使用模拟结果更新顶点的 Z 轴坐标
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            vertices[index].co.z = h[i, j]
            #print(h[grid_size % 2,grid_size % 2])

    # 插入形状关键帧
    for frame in range(t_steps):
        simulation_key.value = 1.0 if frame == t else 0.0
        print(h[grid_size % 2,grid_size % 2])
        simulation_key.keyframe_insert(data_path="value", frame=frame + 1)

# 设置摄像机和光源位置
bpy.data.objects['Camera'].location = (grid_size / 2, -grid_size, grid_size / 2)
bpy.data.objects['Camera'].rotation_euler = (math.radians(60), 0, math.radians(90))
bpy.data.objects['Light'].location = (grid_size / 2, -grid_size / 2, grid_size)