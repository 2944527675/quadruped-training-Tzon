import time
import numpy as np
import mujoco
import mujoco.viewer


# =========================
# 加载模型
# =========================
model = mujoco.MjModel.from_xml_path(
    "black_description/black_description.xml"
)
# 关闭所有 actuator 的作用
model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

# Viewer 的内置 Reset 会恢复 model.qpos0，因此把自定义初始姿态写入这里，
# 确保程序启动和按 Reset 使用完全相同的状态。
model.qpos0[:] = np.array([
    0.0, 0.0, 0.6,                    # base position
    1.0, 0.0, 0.0, 0.0,                # base quaternion (w, x, y, z)
    0, 0, 0,                # FL
    0, 0, 0,                   # FR
    0, 0, 0,                   # RL
    0, 0, 0                   # RR
])

data = mujoco.MjData(model)

# =========================
# 设置初始状态
# =========================
def reset_state(model, data):

    # 恢复 model.qpos0，并清零速度
    mujoco.mj_resetData(model, data)

    # 根据新的 qpos 重新计算
    mujoco.mj_forward(model, data)


# =========================
# 初始化
# =========================
reset_state(model, data)


# =========================
# 启动 MuJoCo Viewer
# =========================
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        step_start = time.time()

        # 仿真一步
        mujoco.mj_step(model, data)

        # 更新画面
        viewer.sync()

        # 实时运行
        time_left = model.opt.timestep - (time.time() - step_start)

        if time_left > 0:
            time.sleep(time_left)
