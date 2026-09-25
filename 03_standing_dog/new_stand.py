import time
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# 加载模型
# ============================================================

model = mujoco.MjModel.from_xml_path(
    "black_description/black_description.xml"
)

data = mujoco.MjData(model)


# ============================================================
# 站立目标角度
#
# 每条腿：
# hip = 0
# thigh = 0.8
# calf = -1.5
# ============================================================

q_stand = np.array([
     0.0,  0.8, -1.5,    # FL
     0.0,  -0.8, 1.5,    # FR
     0.0,  -0.8, 1.5,    # RR
     0.0,  0.8, -1.5     # RL
])


# ============================================================
# 设置初始关节角
# ============================================================

data.qpos[7:19] = q_stand
data.qvel[:] = 0

# 让 MuJoCo 根据当前姿态重新计算动力学
mujoco.mj_forward(model, data)


# ============================================================
# PD 参数（原来写在 XML <position> 里的 kp / kv，现在搬到 Python）
#
# hip:   kp = 100, kv = 30
# thigh: kp = 60,  kv = 15
# calf:  kp = 60,  kv = 15
#
# 12 个执行器顺序: FL_hip, FL_thigh, FL_calf, FR_*, RR_*, RL_*
# ============================================================

kp_stand = np.tile([100.0, 60.0, 60.0], 4)     # 5 s 之后使用的站立 kp
kv_stand = np.tile([30.0, 15.0, 15.0], 4)      # 5 s 之后使用的站立 kv

TAU_LIMIT = 60.0    # 与原 XML forcerange="-60 60" 一致

# 仿真前 5 s kp、kv 全为 0（关节力矩恒为 0，纯自由运动），5 s 后启用上面的 PD
T_START = 0.5
kp = np.zeros_like(kp_stand)    # 当前生效的 kp，运行时动态切换
kv = np.zeros_like(kv_stand)    # 当前生效的 kv，运行时动态切换

# qpos[0:7] 是浮动基座(freejoint)，其后 12 个是腿关节，顺序与执行器一致
q_index = np.arange(7, 19)
qd_index = np.arange(6, 18)


def pd_torque(q_des, q, qd):
    """位置伺服的等价形式: tau = kp * (q_des - q) - kv * qd，再限幅"""
    tau = kp * (q_des - q) - kv * qd
    return np.clip(tau, -TAU_LIMIT, TAU_LIMIT)


# ============================================================
# 仿真步长
#
# kp/kv 写在 XML <position> 里时，MuJoCo 会用 implicitfast 把阻尼项
# 隐式处理，所以 dt = 2 ms 也稳定。
# 现在 -kv*qd 是 Python 显式算出来的，dt 太大就会数值发散
# （实测：5e-4 发散，3e-4 勉强，<= 2e-4 稳定），所以取 dt = 1e-4。
# 每个界面刷新周期做 SUBSTEPS 次小步，刷新节奏仍是 2 ms。
# ============================================================

DT = 1e-4
SUBSTEPS = 20                   # 20 * 1e-4 s = 2 ms

model.opt.timestep = DT


# ============================================================
# Viewer
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # 每个刷新周期走 SUBSTEPS 个小步，每小步重新算一次 PD
        for _ in range(SUBSTEPS):

            # 前 5 s kp、kv 全为 0（tau = 0）；5 s 后切换为站立 PD 参数
            kp[:] = kp_stand if data.time >= T_START else 0.0
            kv[:] = kv_stand if data.time >= T_START else 0.0

            # 读取关节角 / 关节角速度，在 Python 里算出 PD 力矩写入 data.ctrl
            q = data.qpos[q_index]
            qd = data.qvel[qd_index]
            data.ctrl[:] = pd_torque(q_stand, q, qd)

            # MuJoCo 仿真
            mujoco.mj_step(model, data)

        # 打印 4 个髋关节的当前角度和实际力矩
        # (actuator_force 索引 0/3/6/9 就是 FL/FR/RR/RL 的髋关节电机)
        print(
            "q_hip =",
            data.qpos[[7, 10, 13, 16]],
            "tau_hip =",
            data.actuator_force[[0, 3, 6, 9]]
        )

        # 更新窗口
        viewer.sync()

        time.sleep(SUBSTEPS * DT)